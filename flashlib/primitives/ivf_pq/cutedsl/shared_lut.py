"""Efficient shared-memory ADC-LUT IVF-PQ fine scan (CuTe DSL, SM90).

The classic cuVS / FAISS asymmetric-distance LUT algorithm, executed so it
is actually fast at high ``D`` (where decode+GEMM loses because its
per-candidate cost scales with ``D`` while a LUT scan costs only ``m``
lookups, independent of ``D``).
In the small-``m`` / low-recall regime this beats a full IVF-Flat scan in
raw ms on GIST (D=960), which decode+GEMM never does.

Three things make it fast (each fixed a measured bottleneck, in order):

  1. **Precomputed-table LUT (kills the build).** With residual encoding the
     naive LUT rebuilds ``‖(q-c_L)[s]-cb[s,j]‖²`` -- an ``m·256·dsub`` SIMT
     loop that re-reads the codebook -- for *every* probed list, which
     dominated runtime. Instead decompose (FAISS precomputed tables):
     ``‖(q-c_L)-xhat‖² = ‖q-c_L‖² + Σ_s(‖cb[s,j]‖² + 2⟨c_L[s],cb[s,j]⟩
     - 2⟨q[s],cb[s,j]⟩)``. The ``dsub`` cross terms become a per-query GEMM
     (``qterm2``) and a cached per-index GEMM (``cterm2``) on tensor cores,
     so the device build is a single fp16 subtract per ``(s,j)``. The
     per-list offset ``‖q-c_L‖²`` is added back so partials rank across the
     query's lists.
  2. **fp16 LUT + SMEM code staging.** The table is stored fp16 (fp32
     accumulation), halving SMEM so more CTAs stay resident.
  3. **One query per CTA + warp-shuffle top-k (kills the fold).** The old
     design tiled ``BN`` queries per CTA and folded each chunk's scores into
     the top-k with *one thread per query*, serialising the whole list onto
     a single lane (52% of cycles were barrier stalls waiting on it). Now
     each CTA owns one query: every thread keeps a register top-k over its
     strided candidates with no scan-time barriers, then an iterative
     ``shuffle_sync_bfly`` block arg-min merges the ``T`` local lists.
  4. **Exact re-rank.** The fp16 LUT only ranks the *selection*; the host
     re-ranks an oversampled pool with the exact ADC ``‖rq-xhat‖²`` (shared
     with the decode+GEMM path) so returned distances are ADC-exact.

One CTA per ``(list, query)``.
"""
from __future__ import annotations

import weakref
from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
import cutlass.utils as utils
import cuda.bindings.driver as cuda

from flashlib.primitives.knn.triton._common import _next_pow2
from flashlib.primitives.ivf_pq.cutedsl.fine_scan_host import (
    build_inverse_map,
    reduce_rerank,
)


_INF = 3.4e38


class HopperIvfPqSharedLut:
    """fp16 precomputed-table ADC-LUT fine-scan kernel (SM90, SIMT).

    Config (baked at compile time):
        BN: queries per CTA -- always 1 (the warp-shuffle top-k owns one
            query's candidates), kept as a field for the launch grouping.
        M: number of sub-quantizers (``m``).
        DSUB: sub-vector dimension; KSUB: sub-centroids (256).
        DP: padded working dim (``M * DSUB``); TOPK_PAD: padded top-k width.
        by_residual: residual vs direct PQ encoding.
        threads, nlist, max_qtiles: launch config.
    """

    lut_dtype = cutlass.Float16

    def __init__(
        self, *, BN, M, DSUB, KSUB, DP, TOPK_PAD,
        by_residual, threads, nlist, max_qtiles,
    ):
        assert BN == 1, "one query per CTA (warp-shuffle top-k owns one query)"
        self.BN = BN
        self.M = M
        self.DSUB = DSUB
        self.KSUB = KSUB
        self.DP = DP
        self.TOPK_PAD = TOPK_PAD
        self.by_residual = bool(by_residual)
        self.threads = threads
        self.nlist = nlist
        self.max_qtiles = max_qtiles

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mCent: cute.Tensor, mSortedQid: cute.Tensor,
        mQOff: cute.Tensor, mListOff: cute.Tensor, mCodes: cute.Tensor,
        mCT: cute.Tensor, mQT: cute.Tensor, mPV: cute.Tensor, mPI: cute.Tensor,
        stream: cuda.CUstream,
    ):
        BN, M, KSUB, DP = self.BN, self.M, self.KSUB, self.DP
        TOPK_PAD, T = self.TOPK_PAD, self.threads
        POOL = T * TOPK_PAD
        NWARP = T // 32

        @cute.struct
        class SharedStorage:
            sRQ: cute.struct.MemRange[cutlass.Float32, BN * DP]
            sLUT: cute.struct.MemRange[self.lut_dtype, BN * M * KSUB]
            sQid: cute.struct.MemRange[cutlass.Int32, BN]
            sQRSQ: cute.struct.MemRange[cutlass.Float32, BN]
            sPoolV: cute.struct.MemRange[cutlass.Float32, POOL]
            sPoolI: cute.struct.MemRange[cutlass.Int32, POOL]
            sWarpV: cute.struct.MemRange[cutlass.Float32, NWARP]
            sWarpP: cute.struct.MemRange[cutlass.Int32, NWARP]
            sOutV: cute.struct.MemRange[cutlass.Float32, TOPK_PAD]
            sOutI: cute.struct.MemRange[cutlass.Int32, TOPK_PAD]

        self.shared_storage = SharedStorage

        grid = (self.nlist, self.max_qtiles, 1)
        self.kernel(
            mQ, mCent, mSortedQid, mQOff, mListOff, mCodes, mCT, mQT, mPV, mPI,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mCent: cute.Tensor, mSortedQid: cute.Tensor,
        mQOff: cute.Tensor, mListOff: cute.Tensor, mCodes: cute.Tensor,
        mCT: cute.Tensor, mQT: cute.Tensor, mPV: cute.Tensor, mPI: cute.Tensor,
    ):
        BN, M = self.BN, self.M
        KSUB, DP, TOPK_PAD = self.KSUB, self.DP, self.TOPK_PAD
        T = self.threads
        M4 = (M // 4) * 4

        pid_c, pid_qt, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        qstart = mQOff[pid_c]
        qcount = mQOff[pid_c + 1] - qstart
        tile_base = pid_qt * BN
        c_start = mListOff[pid_c]
        c_end = mListOff[pid_c + 1]

        if tile_base < qcount:
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            sRQ = storage.sRQ.get_tensor(cute.make_layout(BN * DP))
            sLUT = storage.sLUT.get_tensor(cute.make_layout(BN * M * KSUB))
            sQid = storage.sQid.get_tensor(cute.make_layout(BN))
            sQRSQ = storage.sQRSQ.get_tensor(cute.make_layout(BN))
            sPoolV = storage.sPoolV.get_tensor(cute.make_layout(T * TOPK_PAD))
            sPoolI = storage.sPoolI.get_tensor(cute.make_layout(T * TOPK_PAD))
            sWarpV = storage.sWarpV.get_tensor(cute.make_layout(T // 32))
            sWarpP = storage.sWarpP.get_tensor(cute.make_layout(T // 32))
            sOutV = storage.sOutV.get_tensor(cute.make_layout(TOPK_PAD))
            sOutI = storage.sOutI.get_tensor(cute.make_layout(TOPK_PAD))

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

            # ── residual query tile sRQ (BN, DP) ───────────────────────
            i = tid
            while i < BN * DP:
                n = i // DP
                d = i % DP
                val = mQ[sQid[n], d]
                if cutlass.const_expr(self.by_residual):
                    val = val - mCent[pid_c, d]
                sRQ[i] = val
                i += T
            cute.arch.sync_threads()

            # ── per-query offset ‖q-c_L‖² (added back to every score) ───
            # The precomputed-table LUT drops this constant, so we re-add
            # it so partials are comparable across the query's lists.
            if tid < BN:
                base_rq = tid * DP
                acc = cutlass.Float32(0.0)
                d = 0
                while d < DP:
                    v = sRQ[base_rq + d]
                    acc = acc + v * v
                    d += 1
                sQRSQ[tid] = acc
            cute.arch.sync_threads()

            # ── build the fp16 SMEM LUT from precomputed tables ────────
            # sLUT[n,s,j] = cterm[list,s,j] - qterm[q,s,j]
            #            = (‖cb[s,j]‖²+2⟨c_L[s],cb[s,j]⟩) - 2⟨q[s],cb[s,j]⟩
            # so Σ_s sLUT[s,code_s] + ‖q-c_L‖² == ‖(q-c_L)-xhat‖² (ADC).
            # The dsub cross term is folded into mCT/mQT by a per-query /
            # per-index GEMM, so the per-probe build is just a subtract.
            i = tid
            while i < BN * M * KSUB:
                j = i % KSUB
                s = (i // KSUB) % M
                n = i // (KSUB * M)
                qid = sQid[n]
                val = mCT[pid_c, s, j] - mQT[qid, s, j]
                sLUT[i] = val.to(self.lut_dtype)
                i += T
            cute.arch.sync_threads()

            # ── per-thread local top-k over strided candidates ─────────
            # Each thread owns candidates {c_start+tid, +T, ...} and keeps a
            # register top-k; threads never sync during the scan (the old
            # per-chunk fold serialised the whole list onto one thread, which
            # dominated runtime). One query per CTA (BN==1), so lut_base==0.
            lv = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Float32)
            li = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Int32)
            for t in cutlass.range_constexpr(TOPK_PAD):
                lv[t] = cutlass.Float32(_INF)
                li[t] = cutlass.Int32(-1)
            worst = cutlass.Float32(_INF)
            qrsq0 = sQRSQ[0]

            r = c_start + tid
            while r < c_end:
                a0 = cutlass.Float32(0.0)
                a1 = cutlass.Float32(0.0)
                a2 = cutlass.Float32(0.0)
                a3 = cutlass.Float32(0.0)
                s = 0
                while s < M4:
                    c0 = cutlass.Int32(mCodes[r, s])
                    c1 = cutlass.Int32(mCodes[r, s + 1])
                    c2 = cutlass.Int32(mCodes[r, s + 2])
                    c3 = cutlass.Int32(mCodes[r, s + 3])
                    a0 = a0 + sLUT[s * KSUB + c0].to(cutlass.Float32)
                    a1 = a1 + sLUT[(s + 1) * KSUB + c1].to(cutlass.Float32)
                    a2 = a2 + sLUT[(s + 2) * KSUB + c2].to(cutlass.Float32)
                    a3 = a3 + sLUT[(s + 3) * KSUB + c3].to(cutlass.Float32)
                    s += 4
                while s < M:
                    cr = cutlass.Int32(mCodes[r, s])
                    a0 = a0 + sLUT[s * KSUB + cr].to(cutlass.Float32)
                    s += 1
                d = a0 + a1 + a2 + a3 + qrsq0
                if d < worst:
                    pend_d = d
                    pend_i = cutlass.Int32(r)
                    for t in cutlass.range_constexpr(TOPK_PAD):
                        cur_d = lv[t]
                        cur_i = li[t]
                        take = cur_d > pend_d
                        lv[t] = pend_d if take else cur_d
                        li[t] = pend_i if take else cur_i
                        pend_d = cur_d if take else pend_d
                        pend_i = cur_i if take else pend_i
                    worst = lv[TOPK_PAD - 1]
                r += T

            # ── publish local top-k to the SMEM merge pool ─────────────
            for t in cutlass.range_constexpr(TOPK_PAD):
                sPoolV[tid * TOPK_PAD + t] = lv[t]
                sPoolI[tid * TOPK_PAD + t] = li[t]
            cute.arch.sync_threads()

            # ── iterative block arg-min: extract TOPK_PAD best from pool ─
            # Each round: every thread takes a local min over its slice, the
            # warp reduces it with shuffles, one thread folds the NWARP warp
            # minima and pops the winner (mark its slot +INF). Balanced work,
            # only 2 block barriers per extracted element.
            lane = tid % 32
            wid = tid // 32
            POOL = T * TOPK_PAD
            for it in cutlass.range_constexpr(TOPK_PAD):
                bv = cutlass.Float32(_INF)
                bp = cutlass.Int32(-1)
                j = tid
                while j < POOL:
                    pv = sPoolV[j]
                    sm = pv < bv
                    bv = pv if sm else bv
                    bp = cutlass.Int32(j) if sm else bp
                    j += T
                off = 16
                while off > 0:
                    ov = cute.arch.shuffle_sync_bfly(bv, off)
                    op = cute.arch.shuffle_sync_bfly(bp, off)
                    sm = ov < bv
                    bv = ov if sm else bv
                    bp = op if sm else bp
                    off //= 2
                if lane == 0:
                    sWarpV[wid] = bv
                    sWarpP[wid] = bp
                cute.arch.sync_threads()
                if tid == 0:
                    fv = sWarpV[0]
                    fp = sWarpP[0]
                    for w in cutlass.range_constexpr(1, T // 32):
                        wv = sWarpV[w]
                        sm = wv < fv
                        fv = wv if sm else fv
                        fp = sWarpP[w] if sm else fp
                    sOutV[it] = fv
                    if fp >= 0:
                        sOutI[it] = sPoolI[fp]
                        sPoolV[fp] = cutlass.Float32(_INF)
                    else:
                        sOutI[it] = cutlass.Int32(-1)
                cute.arch.sync_threads()

            # ── write the (1, TOPK_PAD) partial for this (query, list) ──
            if tid < TOPK_PAD:
                pair_pos = qstart + tile_base
                mPV[pair_pos, tid] = sOutV[tid]
                mPI[pair_pos, tid] = sOutI[tid]


_kernel_cache: dict = {}
_cterm_cache: dict = {}


def _compute_qterm2(Qp: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Per-query LUT cross term ``qterm2[q,s,j] = 2·⟨q[s], cb[s,j]⟩``.

    A batched (over sub-quantizers) GEMM ``(m, nq, dsub) @ (m, dsub, ksub)`` --
    the expensive ``dsub`` reduction now runs once per query on tensor cores
    instead of once per (query, probe) in a SIMT loop. Returns ``(nq, m, ksub)``
    fp32.
    """
    m, ksub, dsub = codebooks.shape
    nq = Qp.shape[0]
    q3 = Qp.view(nq, m, dsub).permute(1, 0, 2).contiguous()      # (m, nq, dsub)
    cb_t = codebooks.permute(0, 2, 1).contiguous()               # (m, dsub, ksub)
    qcb = torch.bmm(q3, cb_t)                                     # (m, nq, ksub)
    return qcb.mul_(2.0).permute(1, 0, 2).contiguous()           # (nq, m, ksub)


def _get_cterm2(
    centroids: torch.Tensor, codebooks: torch.Tensor, nlist: int, by_residual: bool,
) -> torch.Tensor:
    """Per-index LUT term ``cterm2[L,s,j] = ‖cb[s,j]‖² + 2·⟨c_L[s], cb[s,j]⟩``
    (residual) or just ``‖cb[s,j]‖²`` (direct). Index-only, so cache it.

    Keyed by storage address but *validated* by weakref identity: a freed
    index whose address is later reused by a new (same-shape) index must not
    return the stale table, so we recompute unless the cached entry's centroid
    and codebook objects are still the exact tensors passed in. Returns
    ``(nlist, m, ksub)`` fp32."""
    key = (centroids.data_ptr(), codebooks.data_ptr(), nlist,
           int(codebooks.shape[0]), bool(by_residual))
    ent = _cterm_cache.get(key)
    if ent is not None:
        out, cent_ref, cb_ref = ent
        if cent_ref() is centroids and cb_ref() is codebooks:
            return out
    m, ksub, dsub = codebooks.shape
    base = (codebooks * codebooks).sum(-1)                        # (m, ksub)
    if by_residual:
        c3 = centroids.view(nlist, m, dsub).permute(1, 0, 2).contiguous()
        cb_t = codebooks.permute(0, 2, 1).contiguous()           # (m, dsub, ksub)
        ccb = torch.bmm(c3, cb_t)                                 # (m, nlist, ksub)
        out = (base[:, None, :] + ccb.mul_(2.0)).permute(1, 0, 2).contiguous()
    else:
        out = base[None].expand(nlist, m, ksub).contiguous()
    try:
        _cterm_cache[key] = (out, weakref.ref(centroids), weakref.ref(codebooks))
    except TypeError:
        pass
    return out


def _to_cute(t: torch.Tensor):
    mt = cute_rt.from_dlpack(t, assumed_align=16)
    return mt.mark_layout_dynamic(leading_dim=t.ndim - 1)


def ivf_pq_fine_scan_shared_lut(
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    k: int,
    *,
    by_residual: bool,
    threads: int = 128,
    over: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp16 precomputed-table ADC-LUT fine scan (CuTe DSL), then exact re-rank.

    Returns ``(vals, pos)`` -- ``vals`` ``(nq, k)`` ADC-exact squared-L2
    (fp32), ``pos`` ``(nq, k)`` int64 stored-row positions (``-1`` padded).
    ``threads`` is the CTA width (128 is best on H100); each CTA scans one
    query's candidates for one probed list.
    """
    assert Qp.is_cuda and codes.is_cuda
    nq, Dp = Qp.shape
    nprobe = probed.shape[1]
    nlist = list_offsets.shape[0] - 1
    m = codes.shape[1]
    ksub, dsub = codebooks.shape[1], codebooks.shape[2]
    device = Qp.device
    # One query per CTA: the parallel warp-shuffle top-k owns one query's
    # candidates, which keeps the LUT SMEM small (high occupancy) and avoids
    # the serial per-query fold that dominated the multi-query tile design.
    BN = 1

    Qp = Qp.contiguous()
    centroids = centroids.contiguous()
    codebooks = codebooks.contiguous()
    codes = codes.contiguous()

    # Precomputed-table LUT terms (FAISS-style): the dsub cross term is a
    # per-query GEMM (qterm2) + a cached per-index GEMM (cterm2), so the
    # device build is a cheap subtract instead of an m·256·dsub SIMT loop.
    qterm2 = _compute_qterm2(Qp, codebooks)                  # (nq, m, ksub) fp32
    cterm2 = _get_cterm2(centroids, codebooks, nlist, by_residual)

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

    key = (BN, m, dsub, ksub, Dp, TOPK_PAD, bool(by_residual),
           threads, nlist, MAX_QTILES)
    compiled = _kernel_cache.get(key)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compiled is None:
        kernel = HopperIvfPqSharedLut(
            BN=BN, M=m, DSUB=dsub, KSUB=ksub, DP=Dp, TOPK_PAD=TOPK_PAD,
            by_residual=by_residual, threads=threads,
            nlist=nlist, max_qtiles=MAX_QTILES,
        )
        compiled = cute.compile(
            kernel,
            _to_cute(Qp), _to_cute(centroids), _to_cute(sorted_qid),
            _to_cute(q_offsets), _to_cute(list_off_i32), _to_cute(codes),
            _to_cute(cterm2), _to_cute(qterm2), _to_cute(pv_sorted), _to_cute(pi_sorted),
            stream,
        )
        _kernel_cache[key] = compiled

    compiled(
        _to_cute(Qp), _to_cute(centroids), _to_cute(sorted_qid),
        _to_cute(q_offsets), _to_cute(list_off_i32), _to_cute(codes),
        _to_cute(cterm2), _to_cute(qterm2), _to_cute(pv_sorted), _to_cute(pi_sorted),
        stream,
    )

    return reduce_rerank(
        pv_sorted, pi_sorted, perm, nq, nprobe, k,
        Qp=Qp, centroids=centroids, codebooks=codebooks, codes=codes,
        list_offsets=list_offsets, by_residual=by_residual, over=over,
    )


__all__ = ["ivf_pq_fine_scan_shared_lut", "HopperIvfPqSharedLut"]
