"""IVF-PQ fine scan via **decode + WGMMA GEMM** (CuTe DSL, SM90).

The CuTe analogue of the Triton ``"gemm"`` path, and the other half of
the head-to-head the user asked for. Cluster-centric (one CTA per
``(list, query-tile)`` of ``BN`` queries):

  1. **Residual query tile.** Build ``rq = q - centroid_c`` for the tile
     in SMEM as the bf16 WGMMA **A** operand (``BN x Dp``); keep the
     per-query fp32 norm ``‖rq‖²``.
  2. **Decode (no LUT).** Stream the list's PQ codes in ``BM`` chunks and
     *decode* them in-kernel to reconstructed sub-vectors ``xhat`` in
     SMEM (gathering the L2-resident codebook) -- the bf16 WGMMA **B**
     operand (``BM x Dp``) -- shared across the whole query tile; keep
     the per-candidate fp32 norm ``‖xhat‖²``.
  3. **Tensor-core ADC.** ``cross = rq @ xhatᵀ`` via WGMMA into an fp32
     accumulator, then ``dist = ‖rq‖² + ‖xhat‖² - 2·cross`` in registers,
     staged to an on-chip ``(BN, BM)`` score block.
  4. **Top-k.** One thread per query folds its score-block row into a
     register top-k carried across chunks, then writes the partial.

Distances rank with a bf16 cross term, so the host reduce oversamples
and **exact-ADC re-ranks** the pool (shared with the Triton GEMM path)
to make the returned distances ADC-exact.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda

from flashlib.primitives.knn.triton._common import _next_pow2
from flashlib.primitives.ivf_pq.cutedsl.fine_scan_host import (
    build_inverse_map,
    reduce_rerank,
)


_INF = 3.4e38


class HopperIvfPqDecodeGemm:
    """Decode + WGMMA-GEMM IVF-PQ fine-scan kernel (SM90).

    Config baked at compile: BN (queries/CTA, must be 64), BM (codes
    /chunk), M (sub-quantizers), DSUB, KSUB, DP (= M*DSUB, multiple of
    16), TOPK_PAD, by_residual, nlist, max_qtiles.
    """

    def __init__(
        self, *, BN, BM, M, DSUB, KSUB, DP, TOPK_PAD,
        by_residual, nlist, max_qtiles,
    ):
        assert BN == 64, "decode+GEMM uses a 64-row WGMMA tile (BN must be 64)"
        assert DP % 16 == 0, "WGMMA K-dim (Dp) must be a multiple of 16"
        self.BN = BN
        self.BM = BM
        self.M = M
        self.DSUB = DSUB
        self.KSUB = KSUB
        self.DP = DP
        self.TOPK_PAD = TOPK_PAD
        self.by_residual = bool(by_residual)
        self.nlist = nlist
        self.max_qtiles = max_qtiles
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = cutlass.BFloat16
        self.threads = 128
        self.tile_shape_mnk = (BN, BM, DP)

    # ---- WGMMA acc (m,n) TV-layout helpers (from kmeans assign) --------
    @staticmethod
    @cute.jit
    def _layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r

    @cute.jit
    def _layout_acc_mn(self, tiled_mma, acc_layout):
        separated = self._layout_separate(
            tiled_mma.shape_mnk[0], acc_layout[0], tiled_mma.tv_layout_C.stride[1]
        )
        V_M = separated[0]
        V_N = separated[1]
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc_layout[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc_layout[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc_layout[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc_layout[2])
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,          # (nq, DP)        fp32
        mCent: cute.Tensor,       # (nlist, DP)     fp32
        mSortedQid: cute.Tensor,  # (P,)            int32
        mQOff: cute.Tensor,       # (nlist+1,)      int32
        mListOff: cute.Tensor,    # (nlist+1,)      int32
        mCodes: cute.Tensor,      # (Mrows, M)      uint8
        mCB: cute.Tensor,         # (M, KSUB, DSUB) fp32
        mPV: cute.Tensor,         # (P, TOPK_PAD)   fp32
        mPI: cute.Tensor,         # (P, TOPK_PAD)   int32
        stream: cuda.CUstream,
    ):
        BN, BM, DP = self.BN, self.BM, self.DP
        ab_dtype = self.ab_dtype

        q_layout = utils.LayoutEnum.from_tensor(mQ)
        major = q_layout.sm90_mma_major_mode()
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, major, major, self.acc_dtype,
            (1, 1, 1), tiler_mn=(64, BM),
        )
        self.x_smem_layout = sm90_utils.make_smem_layout_a(
            q_layout, self.tile_shape_mnk, ab_dtype, 1,
        )
        self.c_smem_layout = sm90_utils.make_smem_layout_b(
            q_layout, self.tile_shape_mnk, ab_dtype, 1,
        )

        @cute.struct
        class SharedStorage:
            sX: cute.struct.Align[
                cute.struct.MemRange[ab_dtype, cute.cosize(self.x_smem_layout)], 1024]
            sC: cute.struct.Align[
                cute.struct.MemRange[ab_dtype, cute.cosize(self.c_smem_layout)], 1024]
            sScore: cute.struct.MemRange[cutlass.Float32, BN * BM]
            sRQsq: cute.struct.MemRange[cutlass.Float32, BN]
            sXhatSq: cute.struct.MemRange[cutlass.Float32, BM]
            sPartial: cute.struct.MemRange[cutlass.Float32, BM * self.M]
            sQid: cute.struct.MemRange[cutlass.Int32, BN]

        self.shared_storage = SharedStorage

        grid = (self.nlist, self.max_qtiles, 1)
        self.kernel(
            mQ, mCent, mSortedQid, mQOff, mListOff, mCodes, mCB, mPV, mPI,
            self.tiled_mma, self.x_smem_layout, self.c_smem_layout,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mCent: cute.Tensor,
        mSortedQid: cute.Tensor,
        mQOff: cute.Tensor,
        mListOff: cute.Tensor,
        mCodes: cute.Tensor,
        mCB: cute.Tensor,
        mPV: cute.Tensor,
        mPI: cute.Tensor,
        tiled_mma: cute.TiledMma,
        x_smem_layout: cute.ComposedLayout,
        c_smem_layout: cute.ComposedLayout,
    ):
        BN, BM = self.BN, self.BM
        M, DSUB, KSUB, DP = self.M, self.DSUB, self.KSUB, self.DP
        TOPK_PAD = self.TOPK_PAD
        T = self.threads

        pid_c, pid_qt, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        qstart = mQOff[pid_c]
        qend = mQOff[pid_c + 1]
        qcount = qend - qstart
        tile_base = pid_qt * BN
        c_start = mListOff[pid_c]
        c_end = mListOff[pid_c + 1]

        if tile_base < qcount:
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            sX = storage.sX.get_tensor(x_smem_layout.outer, swizzle=x_smem_layout.inner)
            sC = storage.sC.get_tensor(c_smem_layout.outer, swizzle=c_smem_layout.inner)
            sScore = storage.sScore.get_tensor(cute.make_layout(BN * BM))
            sRQsq = storage.sRQsq.get_tensor(cute.make_layout(BN))
            sXhatSq = storage.sXhatSq.get_tensor(cute.make_layout(BM))
            sPartial = storage.sPartial.get_tensor(cute.make_layout(BM * M))
            sQid = storage.sQid.get_tensor(cute.make_layout(BN))

            # ── stage query ids ────────────────────────────────────────
            i = tid
            while i < BN:
                ql = tile_base + i
                qv = cutlass.Int32(0)
                if ql < qcount:
                    qv = mSortedQid[qstart + ql]
                sQid[i] = qv
                i += T
            cute.arch.sync_threads()

            # ── build rq tile (A operand, bf16) ────────────────────────
            i = tid
            total_x = BN * DP
            while i < total_x:
                n = i // DP
                d = i % DP
                qid = sQid[n]
                val = mQ[qid, d]
                if cutlass.const_expr(self.by_residual):
                    val = val - mCent[pid_c, d]
                sX[n, d, 0] = val.to(cutlass.BFloat16)
                i += T
            cute.arch.sync_threads()

            # ── per-query ‖rq‖² (read back the bf16 operand) ───────────
            i = tid
            while i < BN:
                acc = cutlass.Float32(0.0)
                for d in cutlass.range_constexpr(DP):
                    v = sX[i, d, 0].to(cutlass.Float32)
                    acc = acc + v * v
                sRQsq[i] = acc
                i += T
            cute.arch.sync_threads()

            # ── WGMMA partitions / fragments ───────────────────────────
            thr_mma = tiled_mma.get_slice(tid)
            tCsX = thr_mma.partition_A(sX)
            tCsC = thr_mma.partition_B(sC)
            tCrX = tiled_mma.make_fragment_A(tCsX)
            tCrC = tiled_mma.make_fragment_B(tCsC)
            cP = cute.make_identity_tensor((BN, BM))
            ptPcP = thr_mma.partition_C(cP)
            acc = cute.make_rmem_tensor(thr_mma.partition_C(cP).shape, self.acc_dtype)
            acc_mn = cute.make_tensor(acc.iterator, self._layout_acc_mn(tiled_mma, acc.layout))
            ptPcP_mn = cute.make_tensor(
                ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout))
            M_per_thr = cute.size(acc_mn, mode=[0])
            N_per_thr = cute.size(acc_mn, mode=[1])
            num_k_blocks = cute.size(tCrX, mode=[2])

            # ── per-query register top-k ───────────────────────────────
            topv = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Float32)
            topi = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Int32)
            for t in cutlass.range_constexpr(TOPK_PAD):
                topv[t] = cutlass.Float32(_INF)
                topi[t] = cutlass.Int32(-1)
            worst = cutlass.Float32(_INF)
            creg = cute.make_rmem_tensor(cute.make_layout(M), cutlass.Int32)

            list_len = c_end - c_start
            n_chunks = (list_len + BM - 1) // BM
            for ci in cutlass.range(n_chunks, unroll=1):
                chunk_start = c_start + ci * BM

                # decode xhat (B operand, bf16): ONE THREAD PER (row, s)
                # sub-vector. The code byte is read once per sub-vector (vs
                # DSUB times in the per-element form) and the DSUB-contiguous
                # codebook entry mCB[s, code, :] is read as a vectorisable
                # run; ‖xhat‖² is folded in as a per-sub-vector partial so
                # there is no separate SMEM re-read pass.
                i = tid
                n_sub = BM * M
                while i < n_sub:
                    n = i // M
                    s = i % M
                    row = chunk_start + n
                    base = s * DSUB
                    ssum = cutlass.Float32(0.0)
                    if row < c_end:
                        code = cutlass.Int32(mCodes[row, s])
                        for o in cutlass.range_constexpr(DSUB):
                            v = mCB[s, code, o]
                            sC[n, base + o, 0] = v.to(cutlass.BFloat16)
                            ssum = ssum + v * v
                    else:
                        for o in cutlass.range_constexpr(DSUB):
                            sC[n, base + o, 0] = cutlass.BFloat16(0.0)
                    sPartial[i] = ssum
                    i += T
                cute.arch.sync_threads()

                # reduce the M sub-vector partials -> per-candidate ‖xhat‖²
                i = tid
                while i < BM:
                    acc_xs = cutlass.Float32(0.0)
                    for s in cutlass.range_constexpr(M):
                        acc_xs = acc_xs + sPartial[i * M + s]
                    sXhatSq[i] = acc_xs
                    i += T
                cute.arch.sync_threads()

                # WGMMA: cross = rq @ xhatᵀ
                cute.arch.fence_proxy("async.shared", space="cta")
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma, acc,
                        tCrX[(None, None, kb, 0)],
                        tCrC[(None, None, kb, 0)],
                        acc,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                # epilogue: dist = ‖rq‖² + ‖xhat‖² - 2·cross -> sScore
                for ii in cutlass.range_constexpr(M_per_thr):
                    for jj in cutlass.range_constexpr(N_per_thr):
                        m_local = ptPcP_mn[(ii, jj)][0]
                        n_local = ptPcP_mn[(ii, jj)][1]
                        cross = acc_mn[(ii, jj)]
                        dist = sRQsq[m_local] + sXhatSq[n_local] - cutlass.Float32(2.0) * cross
                        if chunk_start + n_local >= c_end:
                            dist = cutlass.Float32(_INF)
                        sScore[m_local * BM + n_local] = dist
                cute.arch.sync_threads()

                # one thread per query folds its score-block row into top-k
                if tid < BN:
                    ql = tile_base + tid
                    if ql < qcount:
                        row_base = tid * BM
                        for b in cutlass.range_constexpr(BM):
                            cd = sScore[row_base + b]
                            if cd < worst:
                                pend_d = cd
                                pend_i = cutlass.Int32(chunk_start + b)
                                for t in cutlass.range_constexpr(TOPK_PAD):
                                    cur_d = topv[t]
                                    cur_i = topi[t]
                                    take = cur_d > pend_d
                                    topv[t] = pend_d if take else cur_d
                                    topi[t] = pend_i if take else cur_i
                                    pend_d = cur_d if take else pend_d
                                    pend_i = cur_i if take else pend_i
                                worst = topv[TOPK_PAD - 1]
                cute.arch.sync_threads()

            if tid < BN:
                ql = tile_base + tid
                if ql < qcount:
                    pair_pos = qstart + ql
                    for t in cutlass.range_constexpr(TOPK_PAD):
                        mPV[pair_pos, t] = topv[t]
                        mPI[pair_pos, t] = topi[t]


class HopperIvfPqDecodeGemmPipelined(HopperIvfPqDecodeGemm):
    """Double-buffered decode + async-WGMMA pipeline (SM90).

    Same math as :class:`HopperIvfPqDecodeGemm`, but the SIMT decode of
    chunk ``ci+1`` is issued *after* the async WGMMA of chunk ``ci`` is
    committed and *before* ``wait_group``, so the codebook gather runs on
    the LSU while the tensor cores crunch the cross term -- the decode
    latency hides behind the GEMM. Needs a 2-stage ``sC`` / ``sXhatSq`` so
    the decode target never aliases the buffer the in-flight WGMMA reads.
    """

    @cute.jit
    def _decode_into(self, sCbuf, sXhatSqbuf, sPartialbuf, chunk_start, c_end,
                     mCodes, mCB):
        """Vectorised decode of one ``BM`` code chunk into ``sCbuf`` (bf16 B
        operand): one thread per ``(row, s)`` sub-vector (code byte hoisted,
        DSUB-contiguous codebook run), folding ``‖xhat‖²`` into ``sXhatSqbuf``
        via per-sub-vector partials. Ends barriered."""
        BM, DSUB = self.BM, self.DSUB
        M = self.M
        T = self.threads
        tid, _, _ = cute.arch.thread_idx()
        i = tid
        n_sub = BM * M
        while i < n_sub:
            n = i // M
            s = i % M
            row = chunk_start + n
            base = s * DSUB
            ssum = cutlass.Float32(0.0)
            if row < c_end:
                code = cutlass.Int32(mCodes[row, s])
                for o in cutlass.range_constexpr(DSUB):
                    v = mCB[s, code, o]
                    sCbuf[n, base + o, 0] = v.to(cutlass.BFloat16)
                    ssum = ssum + v * v
            else:
                for o in cutlass.range_constexpr(DSUB):
                    sCbuf[n, base + o, 0] = cutlass.BFloat16(0.0)
            sPartialbuf[i] = ssum
            i += T
        cute.arch.sync_threads()
        i = tid
        while i < BM:
            acc_xs = cutlass.Float32(0.0)
            for s in cutlass.range_constexpr(M):
                acc_xs = acc_xs + sPartialbuf[i * M + s]
            sXhatSqbuf[i] = acc_xs
            i += T
        cute.arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mCent: cute.Tensor, mSortedQid: cute.Tensor,
        mQOff: cute.Tensor, mListOff: cute.Tensor, mCodes: cute.Tensor,
        mCB: cute.Tensor, mPV: cute.Tensor, mPI: cute.Tensor,
        stream: cuda.CUstream,
    ):
        BN, BM, DP = self.BN, self.BM, self.DP
        ab_dtype = self.ab_dtype

        q_layout = utils.LayoutEnum.from_tensor(mQ)
        major = q_layout.sm90_mma_major_mode()
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, major, major, self.acc_dtype,
            (1, 1, 1), tiler_mn=(64, BM),
        )
        self.x_smem_layout = sm90_utils.make_smem_layout_a(
            q_layout, self.tile_shape_mnk, ab_dtype, 1,
        )
        self.c_smem_layout = sm90_utils.make_smem_layout_b(
            q_layout, self.tile_shape_mnk, ab_dtype, 1,
        )

        @cute.struct
        class SharedStorage:
            sX: cute.struct.Align[
                cute.struct.MemRange[ab_dtype, cute.cosize(self.x_smem_layout)], 1024]
            sC_a: cute.struct.Align[
                cute.struct.MemRange[ab_dtype, cute.cosize(self.c_smem_layout)], 1024]
            sC_b: cute.struct.Align[
                cute.struct.MemRange[ab_dtype, cute.cosize(self.c_smem_layout)], 1024]
            sScore: cute.struct.MemRange[cutlass.Float32, BN * BM]
            sRQsq: cute.struct.MemRange[cutlass.Float32, BN]
            sXhatSq_a: cute.struct.MemRange[cutlass.Float32, BM]
            sXhatSq_b: cute.struct.MemRange[cutlass.Float32, BM]
            sPartial: cute.struct.MemRange[cutlass.Float32, BM * self.M]
            sQid: cute.struct.MemRange[cutlass.Int32, BN]

        self.shared_storage = SharedStorage

        grid = (self.nlist, self.max_qtiles, 1)
        self.kernel(
            mQ, mCent, mSortedQid, mQOff, mListOff, mCodes, mCB, mPV, mPI,
            self.tiled_mma, self.x_smem_layout, self.c_smem_layout,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mCent: cute.Tensor, mSortedQid: cute.Tensor,
        mQOff: cute.Tensor, mListOff: cute.Tensor, mCodes: cute.Tensor,
        mCB: cute.Tensor, mPV: cute.Tensor, mPI: cute.Tensor,
        tiled_mma: cute.TiledMma,
        x_smem_layout: cute.ComposedLayout,
        c_smem_layout: cute.ComposedLayout,
    ):
        BN, BM = self.BN, self.BM
        M, DSUB, KSUB, DP = self.M, self.DSUB, self.KSUB, self.DP
        TOPK_PAD = self.TOPK_PAD
        T = self.threads

        pid_c, pid_qt, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        qstart = mQOff[pid_c]
        qend = mQOff[pid_c + 1]
        qcount = qend - qstart
        tile_base = pid_qt * BN
        c_start = mListOff[pid_c]
        c_end = mListOff[pid_c + 1]

        if tile_base < qcount:
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            sX = storage.sX.get_tensor(x_smem_layout.outer, swizzle=x_smem_layout.inner)
            sC0 = storage.sC_a.get_tensor(c_smem_layout.outer, swizzle=c_smem_layout.inner)
            sC1 = storage.sC_b.get_tensor(c_smem_layout.outer, swizzle=c_smem_layout.inner)
            sScore = storage.sScore.get_tensor(cute.make_layout(BN * BM))
            sRQsq = storage.sRQsq.get_tensor(cute.make_layout(BN))
            sXhatSq0 = storage.sXhatSq_a.get_tensor(cute.make_layout(BM))
            sXhatSq1 = storage.sXhatSq_b.get_tensor(cute.make_layout(BM))
            sPartial = storage.sPartial.get_tensor(cute.make_layout(BM * M))
            sQid = storage.sQid.get_tensor(cute.make_layout(BN))

            # ── stage query ids ────────────────────────────────────────
            i = tid
            while i < BN:
                ql = tile_base + i
                qv = cutlass.Int32(0)
                if ql < qcount:
                    qv = mSortedQid[qstart + ql]
                sQid[i] = qv
                i += T
            cute.arch.sync_threads()

            # ── build rq tile (A operand, bf16) ────────────────────────
            i = tid
            total_x = BN * DP
            while i < total_x:
                n = i // DP
                d = i % DP
                qid = sQid[n]
                val = mQ[qid, d]
                if cutlass.const_expr(self.by_residual):
                    val = val - mCent[pid_c, d]
                sX[n, d, 0] = val.to(cutlass.BFloat16)
                i += T
            cute.arch.sync_threads()

            # ── per-query ‖rq‖² ────────────────────────────────────────
            i = tid
            while i < BN:
                acc0 = cutlass.Float32(0.0)
                for d in cutlass.range_constexpr(DP):
                    v = sX[i, d, 0].to(cutlass.Float32)
                    acc0 = acc0 + v * v
                sRQsq[i] = acc0
                i += T
            cute.arch.sync_threads()

            # ── WGMMA partitions / fragments (both buffers) ────────────
            thr_mma = tiled_mma.get_slice(tid)
            tCsX = thr_mma.partition_A(sX)
            tCsC0 = thr_mma.partition_B(sC0)
            tCsC1 = thr_mma.partition_B(sC1)
            tCrX = tiled_mma.make_fragment_A(tCsX)
            tCrC0 = tiled_mma.make_fragment_B(tCsC0)
            tCrC1 = tiled_mma.make_fragment_B(tCsC1)
            cP = cute.make_identity_tensor((BN, BM))
            ptPcP = thr_mma.partition_C(cP)
            acc = cute.make_rmem_tensor(thr_mma.partition_C(cP).shape, self.acc_dtype)
            acc_mn = cute.make_tensor(acc.iterator, self._layout_acc_mn(tiled_mma, acc.layout))
            ptPcP_mn = cute.make_tensor(
                ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout))
            M_per_thr = cute.size(acc_mn, mode=[0])
            N_per_thr = cute.size(acc_mn, mode=[1])
            num_k_blocks = cute.size(tCrX, mode=[2])

            # ── per-query register top-k ───────────────────────────────
            topv = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Float32)
            topi = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Int32)
            for t in cutlass.range_constexpr(TOPK_PAD):
                topv[t] = cutlass.Float32(_INF)
                topi[t] = cutlass.Int32(-1)
            worst = cutlass.Float32(_INF)

            list_len = c_end - c_start
            n_chunks = (list_len + BM - 1) // BM

            # ── prologue: decode chunk 0 into buffer 0 ─────────────────
            self._decode_into(sC0, sXhatSq0, sPartial, c_start, c_end, mCodes, mCB)

            for ci in cutlass.range(n_chunks, unroll=1):
                cur0 = (ci % 2) == 0
                chunk_start = c_start + ci * BM

                # issue the async WGMMA on the current (already-decoded) buffer
                cute.arch.fence_proxy("async.shared", space="cta")
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                cute.nvgpu.warpgroup.fence()
                if cur0:
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma, acc,
                            tCrX[(None, None, kb, 0)], tCrC0[(None, None, kb, 0)], acc)
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                else:
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma, acc,
                            tCrX[(None, None, kb, 0)], tCrC1[(None, None, kb, 0)], acc)
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()

                # OVERLAP: decode chunk ci+1 into the OTHER buffer while the
                # WGMMA runs on the tensor cores (the whole point).
                nstart = chunk_start + BM
                if nstart < c_end:
                    if cur0:
                        self._decode_into(sC1, sXhatSq1, sPartial, nstart, c_end,
                                          mCodes, mCB)
                    else:
                        self._decode_into(sC0, sXhatSq0, sPartial, nstart, c_end,
                                          mCodes, mCB)

                cute.nvgpu.warpgroup.wait_group(0)

                # ── epilogue: dist = ‖rq‖² + ‖xhat‖² - 2·cross -> sScore
                for ii in cutlass.range_constexpr(M_per_thr):
                    for jj in cutlass.range_constexpr(N_per_thr):
                        m_local = ptPcP_mn[(ii, jj)][0]
                        n_local = ptPcP_mn[(ii, jj)][1]
                        cross = acc_mn[(ii, jj)]
                        xs = cutlass.Float32(0.0)
                        if cur0:
                            xs = sXhatSq0[n_local]
                        else:
                            xs = sXhatSq1[n_local]
                        dist = sRQsq[m_local] + xs - cutlass.Float32(2.0) * cross
                        if chunk_start + n_local >= c_end:
                            dist = cutlass.Float32(_INF)
                        sScore[m_local * BM + n_local] = dist
                cute.arch.sync_threads()

                # ── top-k fold (one thread per query) ──────────────────
                if tid < BN:
                    ql = tile_base + tid
                    if ql < qcount:
                        row_base = tid * BM
                        for b in cutlass.range_constexpr(BM):
                            cd = sScore[row_base + b]
                            if cd < worst:
                                pend_d = cd
                                pend_i = cutlass.Int32(chunk_start + b)
                                for t in cutlass.range_constexpr(TOPK_PAD):
                                    cur_d = topv[t]
                                    cur_i = topi[t]
                                    take = cur_d > pend_d
                                    topv[t] = pend_d if take else cur_d
                                    topi[t] = pend_i if take else cur_i
                                    pend_d = cur_d if take else pend_d
                                    pend_i = cur_i if take else pend_i
                                worst = topv[TOPK_PAD - 1]
                cute.arch.sync_threads()

            if tid < BN:
                ql = tile_base + tid
                if ql < qcount:
                    pair_pos = qstart + ql
                    for t in cutlass.range_constexpr(TOPK_PAD):
                        mPV[pair_pos, t] = topv[t]
                        mPI[pair_pos, t] = topi[t]


class HopperIvfPqDecodeGemmKT(HopperIvfPqDecodeGemm):
    """K-tiled decode + WGMMA-GEMM IVF-PQ fine-scan kernel (SM90).

    The base :class:`HopperIvfPqDecodeGemm` stages the *full* ``Dp`` query
    tile (``sX = BN x Dp``) and decoded list tile (``sC = BM x Dp``) in
    shared memory. At high ``Dp`` (e.g. GIST ``Dp=960``) ``sX`` alone is
    ~120 KB, so only **one** CTA fits per SM (6.25% occupancy, measured)
    and the latency-bound codebook gather of the decode cannot be hidden.

    This variant **K-tiles the GEMM mainloop**: it streams the contraction
    dim ``Dp`` in ``BK``-wide chunks, so SMEM holds only ``BN x BK`` /
    ``BM x BK`` slices regardless of ``Dp``. Occupancy is restored to many
    CTAs/SM, and the scheduler hides each CTA's decode gather behind other
    CTAs' WGMMA -- the same "decode off the critical path via occupancy"
    win that ``BM=16`` gave at low ``Dp``, but now decoupled from ``Dp``.

    ``‖rq‖²`` (needed for cross-list-comparable partials under residual
    encoding) is computed once in a cheap pre-pass; ``‖xhat‖²`` and the
    WGMMA cross term are accumulated across the ``BK`` chunks.
    """

    def __init__(
        self, *, BN, BM, M, DSUB, KSUB, DP, TOPK_PAD,
        by_residual, nlist, max_qtiles, BK, smem_cb=False,
    ):
        assert BN == 64, "decode+GEMM uses a 64-row WGMMA tile (BN must be 64)"
        assert DP % 16 == 0, "WGMMA K-dim (Dp) must be a multiple of 16"
        assert BK % 16 == 0, "BK (WGMMA K chunk) must be a multiple of 16"
        assert DP % BK == 0, "BK must divide Dp"
        assert BK % DSUB == 0, "BK must be a whole number of sub-quantizers"
        self.BN = BN
        self.BM = BM
        self.M = M
        self.DSUB = DSUB
        self.KSUB = KSUB
        self.DP = DP
        self.BK = BK
        self.SK = BK // DSUB          # sub-quantizers per K chunk
        self.NKC = DP // BK           # number of K chunks
        self.smem_cb = bool(smem_cb)  # stage per-K-chunk codebook in SMEM
        self.cb_slice = self.SK * KSUB * DSUB   # codebook slice elements
        self.TOPK_PAD = TOPK_PAD
        self.by_residual = bool(by_residual)
        self.nlist = nlist
        self.max_qtiles = max_qtiles
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = cutlass.BFloat16
        self.threads = 128
        self.tile_shape_mnk = (BN, BM, BK)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mCent: cute.Tensor, mSortedQid: cute.Tensor,
        mQOff: cute.Tensor, mListOff: cute.Tensor, mCodes: cute.Tensor,
        mCB: cute.Tensor, mPV: cute.Tensor, mPI: cute.Tensor,
        stream: cuda.CUstream,
    ):
        BN, BM, BK = self.BN, self.BM, self.BK
        ab_dtype = self.ab_dtype

        q_layout = utils.LayoutEnum.from_tensor(mQ)
        major = q_layout.sm90_mma_major_mode()
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, major, major, self.acc_dtype,
            (1, 1, 1), tiler_mn=(64, BM),
        )
        self.x_smem_layout = sm90_utils.make_smem_layout_a(
            q_layout, self.tile_shape_mnk, ab_dtype, 1,
        )
        self.c_smem_layout = sm90_utils.make_smem_layout_b(
            q_layout, self.tile_shape_mnk, ab_dtype, 1,
        )

        cb_n = self.cb_slice if self.smem_cb else 1

        @cute.struct
        class SharedStorage:
            sX: cute.struct.Align[
                cute.struct.MemRange[ab_dtype, cute.cosize(self.x_smem_layout)], 1024]
            sC: cute.struct.Align[
                cute.struct.MemRange[ab_dtype, cute.cosize(self.c_smem_layout)], 1024]
            sCB: cute.struct.Align[
                cute.struct.MemRange[ab_dtype, cb_n], 1024]
            sScore: cute.struct.MemRange[cutlass.Float32, BN * BM]
            sRQsq: cute.struct.MemRange[cutlass.Float32, BN]
            sXhatSq: cute.struct.MemRange[cutlass.Float32, BM]
            sPartial: cute.struct.MemRange[cutlass.Float32, BM * self.SK]
            sQid: cute.struct.MemRange[cutlass.Int32, BN]

        self.shared_storage = SharedStorage

        grid = (self.nlist, self.max_qtiles, 1)
        self.kernel(
            mQ, mCent, mSortedQid, mQOff, mListOff, mCodes, mCB, mPV, mPI,
            self.tiled_mma, self.x_smem_layout, self.c_smem_layout,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mCent: cute.Tensor, mSortedQid: cute.Tensor,
        mQOff: cute.Tensor, mListOff: cute.Tensor, mCodes: cute.Tensor,
        mCB: cute.Tensor, mPV: cute.Tensor, mPI: cute.Tensor,
        tiled_mma: cute.TiledMma,
        x_smem_layout: cute.ComposedLayout,
        c_smem_layout: cute.ComposedLayout,
    ):
        BN, BM = self.BN, self.BM
        M, DSUB, KSUB, DP = self.M, self.DSUB, self.KSUB, self.DP
        BK, SK, NKC = self.BK, self.SK, self.NKC
        TOPK_PAD = self.TOPK_PAD
        T = self.threads

        pid_c, pid_qt, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        qstart = mQOff[pid_c]
        qend = mQOff[pid_c + 1]
        qcount = qend - qstart
        tile_base = pid_qt * BN
        c_start = mListOff[pid_c]
        c_end = mListOff[pid_c + 1]

        if tile_base < qcount:
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            sX = storage.sX.get_tensor(x_smem_layout.outer, swizzle=x_smem_layout.inner)
            sC = storage.sC.get_tensor(c_smem_layout.outer, swizzle=c_smem_layout.inner)
            sScore = storage.sScore.get_tensor(cute.make_layout(BN * BM))
            sRQsq = storage.sRQsq.get_tensor(cute.make_layout(BN))
            sXhatSq = storage.sXhatSq.get_tensor(cute.make_layout(BM))
            sPartial = storage.sPartial.get_tensor(cute.make_layout(BM * SK))
            sCB = storage.sCB.get_tensor(
                cute.make_layout(self.cb_slice if self.smem_cb else 1))
            sQid = storage.sQid.get_tensor(cute.make_layout(BN))

            # ── stage query ids ────────────────────────────────────────
            i = tid
            while i < BN:
                ql = tile_base + i
                qv = cutlass.Int32(0)
                if ql < qcount:
                    qv = mSortedQid[qstart + ql]
                sQid[i] = qv
                i += T
            cute.arch.sync_threads()

            # ── WGMMA partitions / fragments (K = BK now) ──────────────
            thr_mma = tiled_mma.get_slice(tid)
            tCsX = thr_mma.partition_A(sX)
            tCsC = thr_mma.partition_B(sC)
            tCrX = tiled_mma.make_fragment_A(tCsX)
            tCrC = tiled_mma.make_fragment_B(tCsC)
            cP = cute.make_identity_tensor((BN, BM))
            ptPcP = thr_mma.partition_C(cP)
            acc = cute.make_rmem_tensor(thr_mma.partition_C(cP).shape, self.acc_dtype)
            acc_mn = cute.make_tensor(acc.iterator, self._layout_acc_mn(tiled_mma, acc.layout))
            ptPcP_mn = cute.make_tensor(
                ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout))
            M_per_thr = cute.size(acc_mn, mode=[0])
            N_per_thr = cute.size(acc_mn, mode=[1])
            num_k_blocks = cute.size(tCrX, mode=[2])

            # ── per-query register top-k ───────────────────────────────
            topv = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Float32)
            topi = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Int32)
            for t in cutlass.range_constexpr(TOPK_PAD):
                topv[t] = cutlass.Float32(_INF)
                topi[t] = cutlass.Int32(-1)
            worst = cutlass.Float32(_INF)

            # ── pre-pass: per-query ‖rq‖² over all K chunks (once) ──────
            i = tid
            while i < BN:
                sRQsq[i] = cutlass.Float32(0.0)
                i += T
            cute.arch.sync_threads()
            for ko in cutlass.range(NKC, unroll=1):
                d0 = ko * BK
                i = tid
                total_x = BN * BK
                while i < total_x:
                    n = i // BK
                    d = i % BK
                    gd = d0 + d
                    qid = sQid[n]
                    val = mQ[qid, gd]
                    if cutlass.const_expr(self.by_residual):
                        val = val - mCent[pid_c, gd]
                    sX[n, d, 0] = val.to(cutlass.BFloat16)
                    i += T
                cute.arch.sync_threads()
                i = tid
                while i < BN:
                    accr = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr(BK):
                        v = sX[i, d, 0].to(cutlass.Float32)
                        accr = accr + v * v
                    sRQsq[i] = sRQsq[i] + accr
                    i += T
                cute.arch.sync_threads()

            # ── candidate-chunk mainloop ───────────────────────────────
            list_len = c_end - c_start
            n_chunks = (list_len + BM - 1) // BM
            for ci in cutlass.range(n_chunks, unroll=1):
                chunk_start = c_start + ci * BM

                i = tid
                while i < BM:
                    sXhatSq[i] = cutlass.Float32(0.0)
                    i += T
                cute.arch.sync_threads()

                # K-loop: refill rq/xhat BK slice, accumulate cross + ‖xhat‖²
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for ko in cutlass.range(NKC, unroll=1):
                    d0 = ko * BK
                    gs0 = d0 // DSUB

                    # rq chunk -> sX (A operand, bf16)
                    i = tid
                    total_x = BN * BK
                    while i < total_x:
                        n = i // BK
                        d = i % BK
                        gd = d0 + d
                        qid = sQid[n]
                        val = mQ[qid, gd]
                        if cutlass.const_expr(self.by_residual):
                            val = val - mCent[pid_c, gd]
                        sX[n, d, 0] = val.to(cutlass.BFloat16)
                        i += T

                    if cutlass.const_expr(self.smem_cb):
                        # stage this K-chunk's codebook slice (SK,KSUB,DSUB)
                        # contiguously into SMEM (coalesced); the per-code
                        # gather then hits SMEM, not scattered L2.
                        i = tid
                        cbsz = SK * KSUB * DSUB
                        kd = KSUB * DSUB
                        while i < cbsz:
                            ls_ = i // kd
                            rem = i % kd
                            code_ = rem // DSUB
                            o_ = rem % DSUB
                            sCB[i] = mCB[gs0 + ls_, code_, o_].to(cutlass.BFloat16)
                            i += T
                        cute.arch.sync_threads()

                    # decode xhat chunk -> sC (B operand, bf16); fold ‖xhat‖²
                    i = tid
                    n_sub = BM * SK
                    while i < n_sub:
                        n = i // SK
                        ls = i % SK
                        row = chunk_start + n
                        base = ls * DSUB
                        ssum = cutlass.Float32(0.0)
                        if row < c_end:
                            if cutlass.const_expr(self.smem_cb):
                                code = cutlass.Int32(mCodes[row, gs0 + ls])
                                cbase = ls * KSUB * DSUB + code * DSUB
                                for o in cutlass.range_constexpr(DSUB):
                                    vb = sCB[cbase + o]
                                    sC[n, base + o, 0] = vb
                                    vf = vb.to(cutlass.Float32)
                                    ssum = ssum + vf * vf
                            else:
                                code = cutlass.Int32(mCodes[row, gs0 + ls])
                                for o in cutlass.range_constexpr(DSUB):
                                    v = mCB[gs0 + ls, code, o]
                                    sC[n, base + o, 0] = v.to(cutlass.BFloat16)
                                    ssum = ssum + v * v
                        else:
                            for o in cutlass.range_constexpr(DSUB):
                                sC[n, base + o, 0] = cutlass.BFloat16(0.0)
                        sPartial[i] = ssum
                        i += T
                    cute.arch.sync_threads()

                    i = tid
                    while i < BM:
                        acc_xs = cutlass.Float32(0.0)
                        for ls in cutlass.range_constexpr(SK):
                            acc_xs = acc_xs + sPartial[i * SK + ls]
                        sXhatSq[i] = sXhatSq[i] + acc_xs
                        i += T
                    cute.arch.sync_threads()

                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.nvgpu.warpgroup.fence()
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma, acc,
                            tCrX[(None, None, kb, 0)],
                            tCrC[(None, None, kb, 0)],
                            acc,
                        )
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    cute.arch.sync_threads()

                # epilogue: dist = ‖rq‖² + ‖xhat‖² - 2·cross -> sScore
                for ii in cutlass.range_constexpr(M_per_thr):
                    for jj in cutlass.range_constexpr(N_per_thr):
                        m_local = ptPcP_mn[(ii, jj)][0]
                        n_local = ptPcP_mn[(ii, jj)][1]
                        cross = acc_mn[(ii, jj)]
                        dist = sRQsq[m_local] + sXhatSq[n_local] - cutlass.Float32(2.0) * cross
                        if chunk_start + n_local >= c_end:
                            dist = cutlass.Float32(_INF)
                        sScore[m_local * BM + n_local] = dist
                cute.arch.sync_threads()

                if tid < BN:
                    ql = tile_base + tid
                    if ql < qcount:
                        row_base = tid * BM
                        for b in cutlass.range_constexpr(BM):
                            cd = sScore[row_base + b]
                            if cd < worst:
                                pend_d = cd
                                pend_i = cutlass.Int32(chunk_start + b)
                                for t in cutlass.range_constexpr(TOPK_PAD):
                                    cur_d = topv[t]
                                    cur_i = topi[t]
                                    take = cur_d > pend_d
                                    topv[t] = pend_d if take else cur_d
                                    topi[t] = pend_i if take else cur_i
                                    pend_d = cur_d if take else pend_d
                                    pend_i = cur_i if take else pend_i
                                worst = topv[TOPK_PAD - 1]
                cute.arch.sync_threads()

            if tid < BN:
                ql = tile_base + tid
                if ql < qcount:
                    pair_pos = qstart + ql
                    for t in cutlass.range_constexpr(TOPK_PAD):
                        mPV[pair_pos, t] = topv[t]
                        mPI[pair_pos, t] = topi[t]


def _pick_bk(DP: int, DSUB: int, cap: int = 128) -> int:
    """Largest K-chunk ``BK`` that divides ``Dp``, is a whole number of
    sub-quantizers, and is a multiple of 16 (WGMMA K). Prefers 64-aligned
    (cleanest bf16 SMEM swizzle), then 32-, then 16-aligned."""
    cap = min(cap, DP)
    for align in (64, 32, 16):
        bk = (cap // align) * align
        while bk >= align:
            if DP % bk == 0 and bk % DSUB == 0 and bk % 16 == 0:
                return bk
            bk -= align
    return DP


_kernel_cache: dict = {}


def _to_cute(t: torch.Tensor):
    mt = cute_rt.from_dlpack(t, assumed_align=16)
    return mt.mark_layout_dynamic(leading_dim=t.ndim - 1)


def ivf_pq_fine_scan_decode_gemm(
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    k: int,
    *,
    by_residual: bool,
    BN: int = 64,
    BM: Optional[int] = None,
    over: int = 2,
    pipeline: bool = False,
    ktile: Optional[bool] = None,
    BK: Optional[int] = None,
    smem_cb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode + WGMMA-GEMM fine scan (CuTe DSL), then exact-ADC re-rank.

    Args:
        BM: candidates per chunk (WGMMA ``N``). ``None`` (default) picks per
            kernel: ``16`` for the serial full-Dp kernel (its 120 KB ``sX``
            already pins occupancy at ~1 CTA/SM, so a small ``sC``/``sScore``
            is all that's free) and ``64`` for the K-tiled kernel (whose tiny
            ``BK``-wide operands leave room for a wide WGMMA ``N`` that
            amortises the per-chunk decode/top-k over more candidates).
        pipeline: if ``True`` use the double-buffered async-WGMMA kernel
            (:class:`HopperIvfPqDecodeGemmPipelined`) that overlaps the
            decode of chunk ``ci+1`` with the WGMMA of chunk ``ci``;
            otherwise the serial :class:`HopperIvfPqDecodeGemm`.
        ktile: K-tile the GEMM mainloop (:class:`HopperIvfPqDecodeGemmKT`)
            so SMEM holds only a ``BK``-wide operand slice instead of the
            full ``Dp``. Decisive at high ``Dp`` (e.g. GIST ``Dp=960``)
            where staging full ``Dp`` collapses occupancy to one CTA/SM.
            ``None`` (default) auto-enables it whenever ``Dp`` is large
            enough that ``_pick_bk`` actually splits the contraction.
        BK: K-chunk width for the K-tiled kernel (auto via ``_pick_bk``).

    Returns ``(vals, pos)`` -- ``vals`` ``(nq, k)`` ADC-exact squared-L2
    (fp32), ``pos`` ``(nq, k)`` int64 stored-row positions (``-1`` pad).
    """
    assert Qp.is_cuda and codes.is_cuda
    nq, Dp = Qp.shape
    nprobe = probed.shape[1]
    nlist = list_offsets.shape[0] - 1
    m = codes.shape[1]
    ksub, dsub = codebooks.shape[1], codebooks.shape[2]
    device = Qp.device

    Qp = Qp.contiguous()
    centroids = centroids.contiguous()
    codebooks = codebooks.contiguous()
    codes = codes.contiguous()

    inv = build_inverse_map(probed, nlist, BN)
    sorted_qid = inv["sorted_qid"]
    q_offsets = inv["q_offsets"]
    perm = inv["perm"]
    P = inv["P"]
    MAX_QTILES = inv["MAX_QTILES"]
    list_off_i32 = list_offsets.contiguous().to(torch.int32)

    TOPK_PAD = _next_pow2(k)
    pv_sorted = torch.full((P, TOPK_PAD), _INF, device=device, dtype=torch.float32)
    pi_sorted = torch.full((P, TOPK_PAD), -1, device=device, dtype=torch.int32)

    if BK is None:
        BK = _pick_bk(Dp, dsub)
    if ktile is None:
        # K-tiling only pays off when the serial kernel's full-Dp operand
        # staging (sX = BN*Dp*2 bytes) collapses occupancy to ~1 CTA/SM AND
        # the decode gather dominates (small dsub = many scattered codebook
        # reads, so hiding that latency via occupancy matters most). When
        # the decode is cheap (large dsub) the K-loop's extra per-chunk
        # operand reloads + syncs cost more than the occupancy buys, so we
        # keep the serial full-Dp kernel. (Measured on GIST: KT wins at
        # m=240/dsub=4, loses at m=60/120.)
        serial_sx = BN * Dp * 2
        ktile = (not pipeline) and (BK < Dp) and (serial_sx >= 96 * 1024) and (dsub <= 4)
    if BM is None:
        # K-tiled wants a wide WGMMA N (BM=64): the K-loop amortises the
        # per-chunk decode/top-k overhead over more candidates and keeps the
        # tensor cores fed. The full-Dp serial kernel wants a small BM=16 so
        # its 120 KB sX still leaves room for >1 resident CTA.
        BM = 64 if ktile else 16

    key = (BN, BM, m, dsub, ksub, Dp, TOPK_PAD, bool(by_residual), nlist,
           MAX_QTILES, bool(pipeline), bool(ktile), BK, bool(smem_cb))
    compiled = _kernel_cache.get(key)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compiled is None:
        if ktile:
            kernel = HopperIvfPqDecodeGemmKT(
                BN=BN, BM=BM, M=m, DSUB=dsub, KSUB=ksub, DP=Dp, TOPK_PAD=TOPK_PAD,
                by_residual=by_residual, nlist=nlist, max_qtiles=MAX_QTILES, BK=BK,
                smem_cb=smem_cb,
            )
        else:
            cls = HopperIvfPqDecodeGemmPipelined if pipeline else HopperIvfPqDecodeGemm
            kernel = cls(
                BN=BN, BM=BM, M=m, DSUB=dsub, KSUB=ksub, DP=Dp, TOPK_PAD=TOPK_PAD,
                by_residual=by_residual, nlist=nlist, max_qtiles=MAX_QTILES,
            )
        compiled = cute.compile(
            kernel,
            _to_cute(Qp), _to_cute(centroids), _to_cute(sorted_qid),
            _to_cute(q_offsets), _to_cute(list_off_i32), _to_cute(codes),
            _to_cute(codebooks), _to_cute(pv_sorted), _to_cute(pi_sorted),
            stream,
        )
        _kernel_cache[key] = compiled

    compiled(
        _to_cute(Qp), _to_cute(centroids), _to_cute(sorted_qid),
        _to_cute(q_offsets), _to_cute(list_off_i32), _to_cute(codes),
        _to_cute(codebooks), _to_cute(pv_sorted), _to_cute(pi_sorted),
        stream,
    )

    return reduce_rerank(
        pv_sorted, pi_sorted, perm, nq, nprobe, k,
        Qp=Qp, centroids=centroids, codebooks=codebooks, codes=codes,
        list_offsets=list_offsets, by_residual=by_residual, over=over,
    )


__all__ = [
    "ivf_pq_fine_scan_decode_gemm",
    "HopperIvfPqDecodeGemm",
    "HopperIvfPqDecodeGemmPipelined",
    "HopperIvfPqDecodeGemmKT",
]
