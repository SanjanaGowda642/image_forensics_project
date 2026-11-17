"""Classical copy-move detector: SIFT keypoints -> g2NN self-matching -> iterative
RANSAC similarity models -> dense pixel verification -> source/target decision.

Pure OpenCV / NumPy, so it works even when PyTorch is not installed.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CloneModel:
    M: np.ndarray                  # 2x3 similarity, maps region A -> region B
    src_pts: np.ndarray            # inlier key-points in A
    dst_pts: np.ndarray            # their partners in B
    mask_a: np.ndarray             # uint8 0/255
    mask_b: np.ndarray
    inliers: int                   # effective evidence count
    kind: str = "kp"               # 'kp' (SIFT+RANSAC) or 'block' (dense block matching)


@dataclass
class Detection:
    forged: bool = False
    models: list = field(default_factory=list)
    target_mask: np.ndarray = None
    source_mask: np.ndarray = None
    arrows: list = field(default_factory=list)       # [((x1,y1),(x2,y2)), ...] source -> target
    n_keypoints: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    region_scores: list = field(default_factory=list)


# --------------------------------------------------------------------------- matching
def _detect_keypoints(gray, max_kp=9000):
    if hasattr(cv2, "SIFT_create"):
        det = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.012, edgeThreshold=15)
        norm = cv2.NORM_L2
    else:                                                    # very old OpenCV builds
        det = cv2.ORB_create(nfeatures=max_kp, scaleFactor=1.2, nlevels=8)
        norm = cv2.NORM_HAMMING
    kps, des = det.detectAndCompute(gray, None)
    if des is None or len(kps) < 8:
        return [], None, norm
    if len(kps) > max_kp:
        order = np.argsort([-k.response for k in kps])[:max_kp]
        kps = [kps[i] for i in order]
        des = des[order]
    return kps, des, norm


def _self_match(kps, des, norm, ratio=0.6, min_dist=24.0, k=8):
    """g2NN matching (Amerini et al.): keep every neighbour that is clearly closer than
    the next one, ignoring spatially-near neighbours (same keypoint, other orientation)."""
    pts = np.array([k_.pt for k_ in kps], np.float32)
    knn = cv2.BFMatcher(norm).knnMatch(des, des, k=min(k, len(kps)))
    pairs = set()
    for i, ms in enumerate(knn):
        cands = [m for m in ms if m.trainIdx != i and np.hypot(*(pts[m.trainIdx] - pts[i])) > min_dist]
        if len(cands) < 2:
            continue
        ds = [m.distance for m in cands]
        for n in range(min(len(cands) - 1, 4)):
            if ds[n] < ratio * ds[n + 1]:
                for m in cands[: n + 1]:
                    j = m.trainIdx
                    pairs.add((min(i, j), max(i, j)))
                break
    if not pairs:
        return pts, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    ij = np.array(sorted(pairs))
    return pts, pts[ij[:, 0]], pts[ij[:, 1]]


def _fit_models(p1, p2, max_models=5, min_inliers=4, reproj=3.0):
    """Iterative RANSAC. Pairs are unordered, so each pair is offered in both directions."""
    remaining = np.ones(len(p1), bool)
    out = []
    for _ in range(max_models):
        idx = np.where(remaining)[0]
        n = len(idx)
        if n < min_inliers:
            break
        A = np.vstack([p1[idx], p2[idx]])
        B = np.vstack([p2[idx], p1[idx]])
        M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=reproj,
                                             maxIters=4000, confidence=0.995)
        if M is None:
            break
        inl = inl.ravel().astype(bool)
        fwd, bwd = inl[:n], inl[n:]
        sel = fwd | bwd
        if sel.sum() < min_inliers:
            break
        src = np.where(fwd[sel, None], p1[idx][sel], p2[idx][sel])
        dst = np.where(fwd[sel, None], p2[idx][sel], p1[idx][sel])
        remaining[idx[sel]] = False
        s = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        if 0.6 < s < 1.7:
            out.append((M, src, dst))
    return out


def _flip_models(gray, kps, des, norm, max_models=2, min_inliers=10, min_dist=24.0):
    """Mirrored copy-move (a region pasted flipped left-right).

    SIFT is not mirror-invariant, so the image is matched against its own horizontal mirror image F.
    A mirrored clone then looks like an ordinary similarity between the image and F; composing that
    with the flip gives the real-image relation  p -> flip(M p)  (a reflection)."""
    if des is None or len(kps) < 8:
        return []
    h, w = gray.shape
    kf, df, _ = _detect_keypoints(cv2.flip(gray, 1))
    if df is None:
        return []
    pa = np.array([k.pt for k in kps], np.float32)
    pf = np.array([k.pt for k in kf], np.float32)
    A, B = [], []
    for ms in cv2.BFMatcher(norm).knnMatch(des, df, k=2):
        if len(ms) == 2 and ms[0].distance < 0.7 * ms[1].distance:
            A.append(pa[ms[0].queryIdx])
            B.append(pf[ms[0].trainIdx])
    if len(A) < min_inliers:
        return []
    A, B = np.array(A, np.float32), np.array(B, np.float32)
    Bi = B.copy()
    Bi[:, 0] = (w - 1) - Bi[:, 0]                          # F coordinates -> real-image coordinates
    far = np.hypot(*(A - Bi).T) > min_dist                 # a point matching its own mirror position is symmetry, not a clone
    A, B, Bi = A[far], B[far], Bi[far]
    out = []
    for _ in range(max_models):
        if len(A) < min_inliers:
            break
        M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                             maxIters=4000, confidence=0.995)
        if M is None:
            break
        inl = inl.ravel().astype(bool)
        if inl.sum() < min_inliers:
            break
        sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        if 0.6 < sc < 1.7:
            Mt = M.copy()                                   # flip o M :  x_out = (w-1) - (M00 x + M01 y + M02)
            Mt[0, :] = -M[0, :]
            Mt[0, 2] = (w - 1) - M[0, 2]
            for src_k, dst_k in _split_sides(A[inl], Bi[inl], min_inliers):
                out.append((Mt, src_k, dst_k))
        A, B, Bi = A[~inl], B[~inl], Bi[~inl]
    return out


def _split_sides(src, dst, min_inliers):
    """A mirror about an axis is its own inverse, so one fit holds points of BOTH copies.
    Split them into the two compact groups so each side is verified on its own."""
    if len(src) < 2 * min_inliers:
        return [(src, dst)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lab, cen = cv2.kmeans(src.astype(np.float32), 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    groups = [lab == 0, lab == 1]
    if min(int(g.sum()) for g in groups) < min_inliers:
        return [(src, dst)]
    spread = [float(np.mean(np.hypot(*(src[g] - cen[i]).T))) for i, g in enumerate(groups)]
    sep = float(np.hypot(*(cen[0] - cen[1])))
    if sep < 1.8 * max(spread):                              # one blob, nothing to split
        return [(src, dst)]
    return [(src[g], dst[g]) for g in groups]


def _block_models(gray, block=16, stride=4, min_dist=32, max_models=3, min_pairs=24, min_verified=40):
    """Dense block matching (translation only). Finds clones of smooth / low-texture content
    where SIFT finds almost no keypoints: 16x16 blocks -> 4x4 pooled signature -> FLANN
    nearest neighbours -> vote on the shift vector."""
    h, w = gray.shape
    if h < block * 3 or w < block * 3:
        return []
    g = gray.astype(np.float32)
    pooled = cv2.blur(g, (4, 4))
    xs = np.arange(0, w - block + 1, stride)
    ys = np.arange(0, h - block + 1, stride)
    cols = [pooled[np.ix_(ys + 2 + 4 * j, xs + 2 + 4 * i)] for j in range(4) for i in range(4)]
    feats = np.stack(cols, -1).reshape(-1, 16).astype(np.float32)
    gx, gy = np.meshgrid(xs, ys)
    px, py = gx.ravel(), gy.ravel()
    hp_signed = g - cv2.GaussianBlur(g, (0, 0), 3.0)
    win = np.lib.stride_tricks.sliding_window_view(hp_signed, (block, block))
    hp = np.abs(hp_signed)
    hp_blk = cv2.blur(hp, (block, block))[np.ix_(ys + block // 2, xs + block // 2)].ravel()
    std_blk = _texture(gray, block)[np.ix_(ys + block // 2, xs + block // 2)].ravel()
    ok = (hp_blk > 1.6) & (std_blk > 4.0)
    ids = np.where(ok)[0]
    if len(ids) < 2 * min_pairs:
        return []
    f = np.ascontiguousarray(feats[ids])
    index = cv2.flann_Index(f, dict(algorithm=1, trees=4))
    nn, dd = index.knnSearch(f, 6, params=dict(checks=48))
    q = np.repeat(np.arange(len(ids)), nn.shape[1])
    n = nn.ravel()
    d2 = dd.ravel()
    sel = (n != q) & (d2 < 220.0) & (n >= 0)
    q, n = q[sel], n[sel]
    qi, ni = ids[q], ids[n]
    dx, dy = px[ni] - px[qi], py[ni] - py[qi]
    far = np.hypot(dx, dy) >= min_dist
    qi, ni, dx, dy = qi[far], ni[far], dx[far], dy[far]
    flip = (dy < 0) | ((dy == 0) & (dx < 0))             # canonical orientation of the shift
    dx, dy = np.where(flip, -dx, dx), np.where(flip, -dy, dy)
    qi, ni = np.where(flip, ni, qi), np.where(flip, qi, ni)
    if len(dx) < min_pairs:
        return []
    bx, by = np.round(dx / stride).astype(int), np.round(dy / stride).astype(int)
    off = 4000
    keys = (bx + off) * 20000 + (by + off)
    uk, cnt = np.unique(keys, return_counts=True)
    table = dict(zip(uk.tolist(), cnt.tolist()))
    order = np.argsort(-cnt)
    out, used = [], []
    for oi in order:
        if len(out) >= max_models:
            break
        k = int(uk[oi])
        kx, ky = k // 20000 - off, k % 20000 - off
        tot = sum(table.get((kx + a + off) * 20000 + (ky + b + off), 0) for a in (-1, 0, 1) for b in (-1, 0, 1))
        if tot < min_pairs:
            break
        if any(abs(kx - ux) <= 2 and abs(ky - uy) <= 2 for ux, uy in used):
            continue
        used.append((kx, ky))
        m = (np.abs(bx - kx) <= 1) & (np.abs(by - ky) <= 1)
        # verify every pair at full resolution: normalised correlation of the high-frequency detail
        sel_i, sel_j = qi[m], ni[m]
        A = win[py[sel_i], px[sel_i]].reshape(len(sel_i), -1)
        B = win[py[sel_j], px[sel_j]].reshape(len(sel_j), -1)
        ncc = (A * B).sum(1) / (np.sqrt((A * A).sum(1) * (B * B).sum(1)) + 1e-6)
        good = ncc > 0.80
        if good.sum() < min_verified:
            continue
        m = np.where(m)[0][good]
        sdx, sdy = float(np.median(dx[m])), float(np.median(dy[m]))
        c = block / 2.0
        src = np.stack([px[qi[m]] + c, py[qi[m]] + c], 1).astype(np.float32)
        dst = np.stack([px[ni[m]] + c, py[ni[m]] + c], 1).astype(np.float32)
        M = np.array([[1, 0, sdx], [0, 1, sdy]], np.float64)
        out.append((M, src, dst))
    return out


# --------------------------------------------------------------------------- dense verification
def _fill_holes(m):
    h, w = m.shape
    ff = np.pad(m, 1)
    cv2.floodFill(ff, None, (0, 0), 255)
    holes = cv2.bitwise_not(ff)[1:-1, 1:-1]
    return cv2.bitwise_or(m, holes)


def _texture(gray, k=7):
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k))
    return np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - m * m, 0))


def _dense_region(img_bgr, M, src_pts, dst_pts, e_min=None):
    """Grow the keypoint evidence into a pixel mask of region A (and B = M(A)).

    * pixel evidence = |I(q) - I(Mq)| is small
    * flat pixels (sky, water, walls) match at ANY shift, so they only count when they sit
      next to textured matching pixels -> the mask follows the real border instead of
      bleeding into smooth surroundings.
    """
    h, w = img_bgr.shape[:2]
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.0).astype(np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    partner = cv2.warpAffine(blur, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((h, w), np.uint8) * 255, M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    Ds = cv2.blur(np.abs(blur - partner).mean(axis=2), (5, 5))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tex = _texture(gray)
    tex_p = cv2.warpAffine(tex, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    xs = np.clip(src_pts[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(src_pts[:, 1].round().astype(int), 0, h - 1)
    if e_min is None:        # how much texture counts as evidence scales with the texture of the matched key-points
        e_min = float(np.clip(0.5 * np.median(tex[ys, xs]), 3.5, 14.0))
    informative = np.minimum(tex, tex_p) > e_min
    T = float(np.clip(2.0 * np.percentile(Ds[ys, xs], 75) + 3.0, 6.0, 30.0))

    hull = cv2.convexHull(src_pts.astype(np.int32))
    hm = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hm, hull, 255)
    area = float(max(cv2.contourArea(hull), 1.0))
    root = float(np.sqrt(area))
    r = int(min(0.35 * root + 8, 120))
    hm = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

    cand = (Ds < T) & valid & (hm > 0)
    core = (cand & informative).astype(np.uint8) * 255
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(core)
    keep = set()
    for x, y in zip(xs, ys):
        win = lab[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        keep.update(int(v) for v in np.unique(win) if v > 0)
    if not keep:
        return None, None, T
    core = np.isin(lab, list(keep)).astype(np.uint8) * 255
    rb = max(5, int(0.07 * root))
    bridge = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rb + 1, 2 * rb + 1)))
    mask_a = (cand & (bridge > 0)).astype(np.uint8) * 255
    k = max(5, int(0.05 * root)) | 1
    mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask_a = _fill_holes(mask_a)
    mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab = cv2.connectedComponents(mask_a)
    keep = {int(lab[y, x]) for x, y in zip(xs, ys) if lab[y, x] > 0}
    if not keep:
        return None, None, T
    mask_a = np.isin(lab, list(keep)).astype(np.uint8) * 255
    mask_b = cv2.warpAffine(mask_a, M, (w, h), flags=cv2.INTER_NEAREST)
    return mask_a, mask_b, T


def _shift(a, dx, dy):
    return np.roll(np.roll(a, dy, 0), dx, 1)


def border_cues(img_bgr, mask):
    """How 'out of place' is a region?  Returns (cross, ctx).

    cross = mean Lab step between neighbouring pixels that straddle the region border.
            A pasted patch meets its new surroundings abruptly; the original blends in.
    ctx   = colour difference between the region and the ring around it.
    (On 23 realistic clone/original pairs `cross` picked the pasted copy every time.)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    near = (cv2.dilate(mask, k) > 0) & ~(cv2.erode(mask, k) > 0)
    steps = []
    for dx, dy in ((1, 0), (0, 1)):
        d = np.abs(lab - _shift(lab, dx, dy)).sum(2)
        steps.append(d[(m != _shift(m, dx, dy)) & near])
    steps = np.concatenate(steps)
    cross = float(steps.mean()) if len(steps) else 0.0
    ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0) & \
           ~(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0)
    ctx = float(np.abs(lab[m].mean(0) - lab[ring].mean(0)).sum()) if ring.sum() > 20 else 0.0
    return cross, ctx


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


# --------------------------------------------------------------------------- public API
def detect_copy_move(img_bgr, min_inliers=8, max_models=5, min_coverage=0.6) -> Detection:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kps, des, norm = _detect_keypoints(gray)
    det = Detection(target_mask=np.zeros((h, w), np.uint8), source_mask=np.zeros((h, w), np.uint8),
                    n_keypoints=len(kps))
    p1 = p2 = np.zeros((0, 2), np.float32)
    if des is not None and len(kps) >= 8:
        _, p1, p2 = _self_match(kps, des, norm)
    det.n_matches = len(p1)
    cands = [(M, src, dst, "kp") for M, src, dst in (_fit_models(p1, p2, max_models, min_inliers) if len(p1) >= min_inliers else [])]
    cands += [(M, src, dst, "kp") for M, src, dst in _flip_models(gray, kps, des, norm)]
    cands += [(M, src, dst, "block") for M, src, dst in _block_models(gray)]
    if not cands:
        return det

    min_area = max(300, int(0.0006 * h * w))
    for M, src, dst, kind in cands:
        ma, mb, _ = _dense_region(img_bgr, M, src, dst)
        if ma is None:
            continue
        aa, ab = int((ma > 0).sum()), int((mb > 0).sum())
        if aa < min_area or ab < min_area:
            continue
        if kind == "block" and (len(src) < 80 or min(aa, ab) < 0.003 * h * w):
            continue                       # dense-only evidence must be strong: it has no keypoint backing
        # a real clone matches pixel-wise over (almost) the whole key-point hull; chance repeats do not
        hm = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hm, cv2.convexHull(src.astype(np.int32)), 255)
        if ((ma > 0) & (hm > 0)).sum() < min_coverage * max(1, (hm > 0).sum()):
            continue
        overlap = int(((ma > 0) & (mb > 0)).sum())
        if overlap > 0.3 * min(aa, ab):                      # region overlaps its own copy -> self-similar texture
            continue
        det.models.append(CloneModel(M, src, dst, ma, mb, len(src) if kind == "kp" else max(8, len(src) // 6), kind))

    if not det.models:
        return det
    # a weak match next to a much stronger one is almost always texture repeating by chance (grass, sky, noise)
    best = max(m.inliers for m in det.models)
    det.models = [m for m in det.models if m.inliers >= 0.2 * best]

    det.forged = True
    det.n_inliers = int(sum(m.inliers for m in det.models))

    # ---- merge the regions of all models into distinct regions -------------------------------
    regions = []                                   # [centroid, mask]
    for m in det.models:
        for mk in (m.mask_a, m.mask_b):
            c = _centroid(mk)
            rad = 0.6 * np.sqrt((mk > 0).sum())
            for r in regions:
                if np.hypot(r[0][0] - c[0], r[0][1] - c[1]) < rad:
                    r[1] = np.maximum(r[1], mk)
                    break
            else:
                regions.append([c, mk.copy()])

    # ---- which region is the genuine original?  (lowest "out of place" score)
    scores = []
    for _, mk in regions:
        cross, ctx = border_cues(img_bgr, mk)
        scores.append(cross + 0.3 * ctx)
    det.region_scores = scores
    src_i = int(np.argmin(scores))

    tgt = np.zeros((h, w), np.uint8)
    src_m = regions[src_i][1].copy()
    for i, (c, mk) in enumerate(regions):
        if i != src_i:
            tgt = np.maximum(tgt, mk)
            det.arrows.append((regions[src_i][0], c))
    det.target_mask = tgt
    det.source_mask = cv2.bitwise_and(src_m, cv2.bitwise_not(tgt))
    return det
