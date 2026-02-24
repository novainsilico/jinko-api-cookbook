"""Helper utilities for plot overlay calibration and display.

Organization:
- LLM/OCR calibration pipeline:
  OCR, axis/tick extraction, token refinement, LLM gating, and axis fitting.
- Boilerplate helpers:
  I/O, schema shaping, plotting overlays, and MIME/file retrieval utilities.
"""

import concurrent.futures
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

# ============================================================================
# LLM/OCR calibration pipeline
# ============================================================================
# This section contains the full chart understanding path:
# image OCR -> tick candidates -> token refinement -> LLM-consistent filtering
# -> robust axis fitting.


@dataclass
class TextDetection:
    text: str
    conf: float
    box: list[tuple[int, int]]
    center: tuple[float, float]


@dataclass
class TickLabel:
    value: float
    text: str
    pixel: tuple[float, float]
    axis: str
    box: list[tuple[int, int]] | None


@dataclass
class PlotRegion:
    bbox: tuple[int, int, int, int]
    ticks: list[TickLabel]


def _to_num_tokens(txt):
    """Extract numeric-like tokens. Handles integers, floats, sci-notation, and simple ranges."""
    t = txt.replace("–", "-").replace("—", "-")
    return re.findall("-?\\d+(?:[\\.,]\\d+)?(?:e[+-]?\\d+)?", t, flags=re.I)


def _center_of_box(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (float(sum(xs)) / 4.0, float(sum(ys)) / 4.0)


def run_easyocr(img_bgr, lang, enhance=True, allowlist=None, scale_up=1.5):
    """Run EasyOCR and return structured detections.

    Optional:
      - enhance: apply CLAHE
      - allowlist: restrict characters (e.g., digits for tick OCR)
      - scale_up: resize before OCR to help tiny text
    """
    proc = img_bgr.copy()
    if enhance:
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        proc = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    if scale_up and scale_up != 1.0:
        proc = cv2.resize(
            proc, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_CUBIC
        )
    reader = easyocr.Reader(lang, gpu=True)
    results = reader.readtext(proc[:, :, ::-1], allowlist=allowlist)
    out = []
    for box, text, conf in results:
        try:
            text_clean = str(text).strip()
            if not text_clean:
                continue
            poly = []
            for x, y in box:
                if scale_up and scale_up != 1.0:
                    x = int(round(x / scale_up))
                    y = int(round(y / scale_up))
                poly.append((int(x), int(y)))
            out.append(
                TextDetection(
                    text=text_clean,
                    conf=float(conf),
                    box=poly,
                    center=_center_of_box(poly),
                )
            )
        except Exception:
            continue
    return out


def _digits_margin_pass(
    img_bgr,
    lang,
    conf_min=0.2,
    scale_up=3.0,
    band_frac=0.15,
    x_axis_y=None,
    y_axis_x=None,
):
    """
    OCR for numeric ticks around axes.

    - For the y-axis: crop a vertical strip around y_axis_x and run EasyOCR 3 times:
      0°, +90° (clockwise), and -90° (counter-clockwise). Map coords back to full image.
    - For the x-axis: crop a horizontal strip around x_axis_y (as before).
    - If an axis anchor is missing, fall back to legacy left/bottom margins.

    Returns a list[TickLabel].
    """
    H, W = img_bgr.shape[:2]
    ticks = []

    def _is_good_num(td):
        if td.conf < conf_min:
            return None
        nums = _to_num_tokens(td.text)
        if not nums:
            return None
        try:
            return float(nums[0].replace(",", "."))
        except Exception:
            return None

    def _dedup_ticks(items, px_tol=8, val_tol=1e-06):
        out = []
        for t in items:
            x, y = t.pixel
            keep = True
            for u in out:
                ux, uy = u.pixel
                if (
                    abs(t.value - u.value) <= val_tol
                    and abs(x - ux) + abs(y - uy) <= px_tol
                ):
                    keep = False
                    break
            if keep:
                out.append(t)
        return out

    def _map_box_from_rot(poly_rot, rot_code, orig_h, orig_w):
        mapped = []
        for u, v in poly_rot:
            if rot_code == cv2.ROTATE_90_CLOCKWISE:
                x = v
                y = orig_h - 1 - u
            elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                x = orig_w - 1 - v
                y = u
            else:
                x, y = (u, v)
            mapped.append((int(x), int(y)))
        return mapped

    def _map_center_from_rot(center, rot_code, orig_h, orig_w):
        x_rot, y_rot = center
        if rot_code == cv2.ROTATE_90_CLOCKWISE:
            x = y_rot
            y = orig_h - 1 - x_rot
        elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            x = orig_w - 1 - y_rot
            y = x_rot
        else:
            x, y = (x_rot, y_rot)
        return (float(x), float(y))

    def _run_ocr_on_roi_with_rotations(roi_bgr, axis, x0_off, y0_off):
        h0, w0 = roi_bgr.shape[:2]

        def _ocr0():
            dets = run_easyocr(
                roi_bgr, lang, enhance=True, allowlist="0123456789.-", scale_up=scale_up
            )
            out = []
            for td in dets:
                val = _is_good_num(td)
                if val is None:
                    continue
                cx, cy = td.center
                box0 = [(px + x0_off, py + y0_off) for px, py in td.box]
                out.append(
                    TickLabel(
                        val, td.text, (cx + x0_off, cy + y0_off), axis=axis, box=box0
                    )
                )
            return out

        def _ocr_cw():
            roi_cw = cv2.rotate(roi_bgr, cv2.ROTATE_90_CLOCKWISE)
            dets = run_easyocr(
                roi_cw, lang, enhance=True, allowlist="0123456789.-", scale_up=scale_up
            )
            out = []
            for td in dets:
                val = _is_good_num(td)
                if val is None:
                    continue
                u, v = td.center
                x_local, y_local = _map_center_from_rot(
                    (u, v), cv2.ROTATE_90_CLOCKWISE, h0, w0
                )
                box_local = _map_box_from_rot(td.box, cv2.ROTATE_90_CLOCKWISE, h0, w0)
                box_full = [(px + x0_off, py + y0_off) for px, py in box_local]
                out.append(
                    TickLabel(
                        val,
                        td.text,
                        (x_local + x0_off, y_local + y0_off),
                        axis=axis,
                        box=box_full,
                    )
                )
            return out

        def _ocr_ccw():
            roi_ccw = cv2.rotate(roi_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            dets = run_easyocr(
                roi_ccw, lang, enhance=True, allowlist="0123456789.-", scale_up=scale_up
            )
            out = []
            for td in dets:
                val = _is_good_num(td)
                if val is None:
                    continue
                u, v = td.center
                x_local, y_local = _map_center_from_rot(
                    (u, v), cv2.ROTATE_90_COUNTERCLOCKWISE, h0, w0
                )
                box_local = _map_box_from_rot(
                    td.box, cv2.ROTATE_90_COUNTERCLOCKWISE, h0, w0
                )
                box_full = [(px + x0_off, py + y0_off) for px, py in box_local]
                out.append(
                    TickLabel(
                        val,
                        td.text,
                        (x_local + x0_off, y_local + y0_off),
                        axis=axis,
                        box=box_full,
                    )
                )
            return out

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            f0 = ex.submit(_ocr0)
            f90 = ex.submit(_ocr_cw)
            f_90 = ex.submit(_ocr_ccw)
            out0 = f0.result()
            out90 = f90.result()
            out_90 = f_90.result()
        all_results = out0 + out90 + out_90
        return _dedup_ticks(all_results, px_tol=8, val_tol=1e-06)

    if y_axis_x is not None:
        band_w_left = int(band_frac * W)
        band_w_right = int(band_frac * W * 0.5)
        x0 = max(0, y_axis_x - band_w_left)
        x1 = min(W, y_axis_x + band_w_right)
        y_roi = img_bgr[:, x0:x1]
        ticks.extend(
            _run_ocr_on_roi_with_rotations(y_roi, axis="y", x0_off=x0, y0_off=0)
        )
    else:
        left_w = int(band_frac * W)
        x0, x1 = (0, left_w)
        y_roi = img_bgr[:, x0:x1]
        ticks.extend(
            _run_ocr_on_roi_with_rotations(y_roi, axis="y", x0_off=x0, y0_off=0)
        )
    if x_axis_y is not None:
        band_h_below = int(band_frac * H)
        band_h_above = int(band_frac * H * 0.5)
        y0 = max(0, x_axis_y - band_h_above)
        y1 = min(H, x_axis_y + band_h_below)
        x_roi = img_bgr[y0:y1, :]
        ticks.extend(
            _run_ocr_on_roi_with_rotations(x_roi, axis="x", x0_off=0, y0_off=y0)
        )
    else:
        bot_h = int(band_frac * H)
        y0, y1 = (H - bot_h, H)
        x_roi = img_bgr[y0:y1, :]
        ticks.extend(
            _run_ocr_on_roi_with_rotations(x_roi, axis="x", x0_off=0, y0_off=y0)
        )
    return ticks


def process_image(
    image_path, lang=None, numeric_conf_min=0.3, x_axis_y=None, y_axis_x=None
):
    """
    takes an image as input and the coordinates of the x axis and y axis

    Goal is to locate the ticks on both axis.
    """
    if lang is None:
        lang = ["en"]
    if isinstance(image_path, str):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_ref = image_path
    else:
        img_bgr = image_path.copy()
        image_ref = "<in_memory_image>"
    H, W = img_bgr.shape[:2]
    plots_json = []
    if x_axis_y is None or y_axis_x is None:
        raise ValueError("margins_only=True requires x_axis_y and y_axis_x")
    ticks = _digits_margin_pass(
        img_bgr, lang, conf_min=numeric_conf_min, x_axis_y=x_axis_y, y_axis_x=y_axis_x
    )
    full_box = (0, 0, W, H)
    plots_json.append(asdict(PlotRegion(bbox=full_box, ticks=ticks)))
    out = {
        "image_path": image_ref,
        "image_size": {"width": int(W), "height": int(H)},
        "plots": plots_json,
    }
    print(out)
    return out


class AxisCal:

    def __init__(self, a, b, axis, mode="linear"):
        self.a, self.b, self.axis, self.mode = (float(a), float(b), axis, mode)

    def _f(self, v):
        v = np.asarray(v, float)
        if self.mode.startswith("log"):
            log_base = float(self.mode[3:])
            v = np.clip(v, 1e-12, None)
            return np.log(v) / np.log(float(log_base))
        return v

    def v2p(self, v):
        v = self._f(v)
        return self.a * v + self.b

    def p2v(self, p):
        p = np.asarray(p, float)
        fv = (p - self.b) / (self.a if self.a != 0 else 1e-12)
        if self.mode.startswith("log"):
            log_base = float(self.mode[3:])
            return np.power(float(log_base), fv)
        return fv


def _detect_axes(
    img,
    h_angle_deg=3.0,
    v_angle_deg=3.0,
    canny_lo=50,
    canny_hi=140,
    min_line_frac=0.1,
    max_gap_px=12,
):
    """
    Detect x and y axes using spatially constrained Hough transforms.

    Returns:
      (x_axis_y, y_axis_x, x_axis_span, y_axis_span)
      for now we do not use x_axis_span, y_axis_span (used for debugging)
      x_axis_y corresponds to the pixel in height where is located the x line (0 being the top of the image), and opposite for y_axis_x
    """
    import cv2
    import math
    import numpy as np

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.bilateralFilter(gray, 5, 20, 20)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    min_len = int(min(H, W) * min_line_frac)

    def parallel_support(x0, y0, x1, y1, band=2):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.linspace(y0, y1, n).astype(int)
        cnt = 0
        for x, y in zip(xs, ys):
            if 0 <= x < W and 0 <= y < H:
                cnt += np.count_nonzero(
                    edges[
                        max(0, y - band) : min(H, y + band + 1),
                        max(0, x - band) : min(W, x + band + 1),
                    ]
                )
        return cnt / max(1, n)

    def perpendicular_density_vertical(x, y0, y1, band=6):
        tot = 0
        for y in range(y0, y1):
            if 0 <= y < H:
                tot += np.count_nonzero(edges[y, max(0, x - band) : min(W, x + band)])
        return tot / max(1, y1 - y0)

    def perpendicular_density_horizontal(y, x0, x1, band=6):
        tot = 0
        for x in range(x0, x1):
            if 0 <= x < W:
                tot += np.count_nonzero(edges[max(0, y - band) : min(H, y + band), x])
        return tot / max(1, x1 - x0)

    def local_noise(x0, y0, x1, y1, pad=8):
        xa = max(0, min(x0, x1) - pad)
        xb = min(W, max(x0, x1) + pad)
        ya = max(0, min(y0, y1) - pad)
        yb = min(H, max(y0, y1) + pad)
        area = (xb - xa) * (yb - ya)
        if area <= 0:
            return 1.0
        return np.count_nonzero(edges[ya:yb, xa:xb]) / area

    left_limit = int(0.4 * W)
    edges_left = edges.copy()
    edges_left[:, left_limit:] = 0

    # returns all the vertical lines
    v_lines = cv2.HoughLinesP(
        edges_left,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min_len,
        maxLineGap=max_gap_px,
    )
    y_axis_x = None
    y_axis_span = None
    best_v_score = 0.0
    if v_lines is not None:
        v_eps = math.tan(math.radians(v_angle_deg))
        for x0, y0, x1, y1 in v_lines.reshape(-1, 4):
            dy = abs(y1 - y0)
            dx = abs(x1 - x0)
            if dy == 0 or dx / dy > v_eps:
                continue
            length = math.hypot(x1 - x0, y1 - y0)
            ps = parallel_support(x0, y0, x1, y1)
            pd = perpendicular_density_vertical(
                int(round((x0 + x1) / 2)), min(y0, y1), max(y0, y1)
            )
            noise = local_noise(x0, y0, x1, y1)
            score = length * ps / ((1.0 + pd) * (1.0 + noise))
            if score > best_v_score:
                best_v_score = score
                y_axis_x = int(round((x0 + x1) / 2))
                y_axis_span = (min(y0, y1), max(y0, y1))
    bottom_limit = int(0.6 * H)
    edges_bottom = edges.copy()
    edges_bottom[:bottom_limit, :] = 0

    # returns all the horizontal lines
    h_lines = cv2.HoughLinesP(
        edges_bottom,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=min_len,
        maxLineGap=max_gap_px,
    )
    x_axis_y = None
    x_axis_span = None
    best_h_score = 0.0
    if h_lines is not None:
        h_eps = math.tan(math.radians(h_angle_deg))
        for x0, y0, x1, y1 in h_lines.reshape(-1, 4):
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            if dx == 0 or dy / dx > h_eps:
                continue
            length = math.hypot(x1 - x0, y1 - y0)
            ps = parallel_support(x0, y0, x1, y1)
            pd = perpendicular_density_horizontal(
                int(round((y0 + y1) / 2)), min(x0, x1), max(x0, x1)
            )
            noise = local_noise(x0, y0, x1, y1)
            score = length * ps / ((1.0 + pd) * (1.0 + noise))
            if score > best_h_score:
                best_h_score = score
                x_axis_y = int(round((y0 + y1) / 2))
                x_axis_span = (min(x0, x1), max(x0, x1))
    if y_axis_x is None:
        y_axis_x = int(0.1 * W)
        y_axis_span = (0, H)
    if x_axis_y is None:
        x_axis_y = int(0.9 * H)
        x_axis_span = (0, W)
    return (x_axis_y, y_axis_x, x_axis_span, y_axis_span)


def normalize_axis_breaks(ticks, breaks):
    """
    Normalize axis breaks.

    Supported break formats:
      - breaks = [lo, hi]                      (single break)
      - breaks = [[lo, hi], [lo2, hi2], ...]   (multiple breaks)

    Endpoints may be str/int/float; they are converted to float.

    Rules:
    - A break is kept ONLY if there are at least 2 ticks strictly below `lo`
      AND at least 2 ticks strictly above `hi`.
    - If either side has <= 1 tick:
        - remove the break
        - remove the isolated tick(s) on that side
    - Ticks remain numeric only.
    """
    if not ticks:
        return {"ticks": [], "breaks": []}

    ticks_f = [float(t) for t in ticks]
    ticks_f.sort()

    break_pairs = []
    if not breaks:
        return {"ticks": ticks_f, "breaks": []}

    if (
        isinstance(breaks, (list, tuple))
        and len(breaks) == 2
        and not isinstance(breaks[0], (list, tuple))
    ):
        break_pairs = [list(breaks)]
    elif isinstance(breaks, (list, tuple)):
        break_pairs = list(breaks)
    else:
        return {"ticks": ticks_f, "breaks": []}

    kept_breaks = []

    for br in break_pairs:
        if not isinstance(br, (list, tuple)) or len(br) != 2:
            continue

        try:
            lo = float(br[0])
            hi = float(br[1])
        except (TypeError, ValueError):
            continue

        if lo >= hi:
            continue

        below = [t for t in ticks_f if t <= lo]
        above = [t for t in ticks_f if t >= hi]

        if len(below) >= 2 and len(above) >= 2:
            kept_breaks.append([lo, hi])
            continue

        if len(below) <= 1:
            ticks_f = above

        if len(above) <= 1:
            ticks_f = below

    return {"ticks": ticks_f, "breaks": kept_breaks}


def normalize_axis_if_needed(axis):
    """
    Normalize axis breaks only if they exist.
    Axis is modified in-place and also returned.
    """
    if not axis or "ticks" not in axis:
        return axis

    breaks = axis.get("break")
    if breaks is None:
        return axis

    if isinstance(breaks, list) and len(breaks) == 0:
        axis["break"] = []
        return axis

    norm = normalize_axis_breaks(axis["ticks"], breaks)
    axis["ticks"] = norm["ticks"]
    axis["break"] = norm["breaks"]
    return axis


def _vandermonde(x, deg):
    return np.vander(x, N=deg + 1, increasing=False)


def _huber_weights(residuals, delta):
    r = np.abs(residuals)
    w = np.ones_like(r, dtype=float)
    mask = r > delta
    w[mask] = delta / (r[mask] + 1e-12)
    return w


def _ols_polyfit_on_set(x, y, deg):
    V = _vandermonde(x, deg)
    beta, *_ = np.linalg.lstsq(V, y, rcond=None)
    return beta


def _irls_huber_on_set(x, y, deg, beta0=None, max_iters=10, delta=None, tol=1e-06):
    """
    IRLS with Huber loss, *only* on the provided (x,y) set.
    Returns beta in np.polyval order (highest degree first).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if beta0 is None:
        beta = _ols_polyfit_on_set(x, y, deg)
    else:
        beta = beta0.copy()
    if delta is None:
        span = max(1.0, float(np.max(y) - np.min(y)))
        delta = 0.03 * span
    for _ in range(max_iters):
        yhat = np.polyval(beta, x)
        r = y - yhat
        w = _huber_weights(r, delta)
        V = _vandermonde(x, deg)
        Wsqrt = np.sqrt(np.clip(w, 1e-12, None))
        VW = V * Wsqrt[:, None]
        yW = y * Wsqrt
        beta_new, *_ = np.linalg.lstsq(VW, yW, rcond=None)
        if np.linalg.norm(beta_new - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new
    return beta


def majority_polyfit(
    x,
    y,
    deg=1,
    n_trials=200,
    residual_tol=5.0,
    alpha=0.5,
    lambda_inlier=1.0,
    gamma_span=0.01,
    random_state=None,
):
    """
    Robust polynomial fit with controlled exploration + axis-aware scoring.

    Additions vs original:
      - Huber loss instead of MSE
      - Score evaluated on the SAME set used for fitting
      - Span reward to avoid degenerate local fits
      - Leave-one-out rescue for single catastrophic outliers
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    N = len(x)
    if N <= deg + 1:
        return np.polyfit(x, y, deg)

    s_min = max(2, deg + 1)
    s_max = N

    best_score = -np.inf
    best_beta = None

    def huber_mean(resid, delta):
        a = np.abs(resid)
        quad = a <= delta
        out = np.empty_like(a, dtype=float)
        out[quad] = 0.5 * a[quad] ** 2
        out[~quad] = delta * (a[~quad] - 0.5 * delta)
        return float(out.mean())

    def score_on(beta, xs, ys):
        yhat = np.polyval(beta, xs)
        resid = ys - yhat
        loss = huber_mean(resid, residual_tol)
        n_used = len(xs)
        span = float(xs.max() - xs.min()) if n_used >= 2 else 0.0
        score = lambda_inlier * n_used - alpha * loss + gamma_span * span
        return score, loss, n_used, resid

    if N <= 10:
        for s in range(s_min, s_max + 1):
            for subset in combinations(range(N), s):
                subset = np.array(subset, int)
                xs, ys = x[subset], y[subset]

                beta0 = _ols_polyfit_on_set(xs, ys, deg)
                yhat = np.polyval(beta0, xs)
                resid = np.abs(ys - yhat)
                if np.any(resid > residual_tol):
                    continue

                score0, _, _, _ = score_on(beta0, xs, ys)
                if score0 > best_score:
                    best_score = score0
                    best_beta = beta0

                beta1 = _irls_huber_on_set(xs, ys, deg, beta0=beta0, max_iters=10)
                score1, _, _, _ = score_on(beta1, xs, ys)
                if score1 > best_score:
                    best_score = score1
                    best_beta = beta1

        if best_beta is None:
            return np.polyfit(x, y, deg)
        return best_beta

    sizes = np.arange(s_min, s_max + 1)
    weights = np.ones_like(sizes, float)
    weights[sizes <= 4] *= 3.0
    probs = weights / weights.sum()

    for _ in range(int(n_trials)):
        s = int(rng.choice(sizes, p=probs))
        subset_idx = np.sort(rng.choice(N, size=s, replace=False))
        xs, ys = x[subset_idx], y[subset_idx]

        beta0 = _ols_polyfit_on_set(xs, ys, deg)
        yhat = np.polyval(beta0, xs)
        resid = np.abs(ys - yhat)
        if np.any(resid > residual_tol):
            continue

        score0, _, _, _ = score_on(beta0, xs, ys)
        if score0 > best_score:
            best_score = score0
            best_beta = beta0

        yhat_all = np.polyval(beta0, x)
        resid_all = np.abs(y - yhat_all)
        expanded_idx = np.where(resid_all <= residual_tol)[0]
        if expanded_idx.size < s_min:
            continue

        xs2, ys2 = x[expanded_idx], y[expanded_idx]
        beta1 = _irls_huber_on_set(xs2, ys2, deg, beta0=beta0, max_iters=10)
        score1, _, _, resid2 = score_on(beta1, xs2, ys2)

        if xs2.size >= s_min + 1:
            worst = np.argmax(np.abs(resid2))
            xs3 = np.delete(xs2, worst)
            ys3 = np.delete(ys2, worst)
            beta2 = _irls_huber_on_set(xs3, ys3, deg, beta0=beta1, max_iters=10)
            score2, _, _, _ = score_on(beta2, xs3, ys3)
            if score2 > score1:
                score1 = score2
                beta1 = beta2

        if score1 > best_score:
            best_score = score1
            best_beta = beta1

    if best_beta is None:
        return np.polyfit(x, y, deg)
    return best_beta


def _extract_crop(img, box, pad=2):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return (img[y0 : y1 + 1, x0 : x1 + 1], (x0, y0))


def _ink_mask(gray):
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(
        th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    return th


def _projection_profile(mask, axis):
    if axis == "x":
        prof = mask.sum(axis=0)
    else:
        prof = mask.sum(axis=1)
    return cv2.blur(prof.astype(np.float32).reshape(-1, 1), (3, 1)).ravel()


def _find_valleys(profile, min_prom_frac=0.05, min_sep_px=2):
    if profile.size < 8:
        return []
    p = (profile - profile.min()) / max(1e-06, profile.max() - profile.min())
    ip = 1.0 - p
    peaks, _ = find_peaks(ip, prominence=min_prom_frac, distance=min_sep_px)
    return peaks.tolist()


def _intensity_centroid(mask):
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] <= 1e-06:
        h, w = mask.shape[:2]
        return (w * 0.5, h * 0.5)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def _is_all_digits_or_sign(s):
    if not isinstance(s, str) or not s:
        return False
    return all((ch.isdigit() or ch in "+-., " for ch in s))


def _closest_in_llm(val, llm_vals, atol_abs, atol_rel):
    if not llm_vals:
        return None
    for lv in llm_vals:
        atol = max(atol_abs, atol_rel * max(1.0, abs(lv)))
        if abs(val - lv) <= atol:
            return lv
    return None


def _tok_quad_or_box(t):
    B = t.get("bbox") or t.get("box") or t.get("rect")
    if B is None:
        return None
    if isinstance(B, (list, tuple)) and len(B) == 4 and isinstance(B[0], (list, tuple)):
        xs = [p[0] for p in B]
        ys = [p[1] for p in B]
        x0, x1 = (min(xs), max(xs))
        y0, y1 = (min(ys), max(ys))
        return (int(x0), int(y0), int(x1), int(y1))
    if isinstance(B, (list, tuple)) and len(B) == 4:
        x0, y0, x1, y1 = map(float, B)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        return (int(x0), int(y0), int(x1), int(y1))
    return None


def _split_merged_digits_with_llm(
    img,
    token,
    axis,
    llm_vals,
    atol_abs=0.25,
    atol_rel=0.02,
    oversize_ratio=1.8,
    min_prom_frac=0.05,
    min_sep_px=2,
):
    """
    If token’s box is oversized (vs typical digit) or text looks run-on (e.g. '9101112'),
    split by projection valleys, then greedily fuse adjacent chunks so that
    concatenated text parses to a number present in llm_vals (within tolerance).
    Each kept chunk gets its intensity centroid and becomes a new tick token.

    Returns: list of refined tokens (could be [token] if no split needed).
    """
    box = _tok_quad_or_box(token)
    if box is None:
        return [token]
    crop, (ox, oy) = _extract_crop(img, box, pad=1)
    if crop.size == 0:
        return [token]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _ink_mask(gray)
    h, w = mask.shape[:2]
    txt = str(token.get("text", ""))
    should_try = False
    if _is_all_digits_or_sign(txt) and len(txt.replace(" ", "")) >= 3:
        should_try = True
    if axis == "x" and w >= oversize_ratio * max(6, h * 0.35):
        should_try = True
    if axis == "y" and h >= oversize_ratio * max(6, w * 0.35):
        should_try = True
    if not should_try:
        return [token]
    prof = _projection_profile(mask, axis)
    cuts = _find_valleys(prof, min_prom_frac=min_prom_frac, min_sep_px=min_sep_px)
    if len(cuts) == 0:
        return [token]
    segs = []
    if axis == "x":
        xs = [0] + cuts + [w - 1]
        for i in range(len(xs) - 1):
            sx, ex = (xs[i], xs[i + 1])
            if ex - sx < 2:
                continue
            segs.append(((sx, 0, ex, h - 1), mask[:, sx:ex]))
    else:
        ys = [0] + cuts + [h - 1]
        for i in range(len(ys) - 1):
            sy, ey = (ys[i], ys[i + 1])
            if ey - sy < 2:
                continue
            segs.append(((0, sy, w - 1, ey), mask[sy:ey, :]))
    if not segs:
        return [token]
    refined = []
    i = 0
    while i < len(segs):
        accepted_j = None
        accepted_val = None
        accepted_centroid = None
        accepted_box_local = None
        for j in range(i + 1, len(segs) + 1):
            if axis == "x":
                x0 = segs[i][0][0]
                x1 = segs[j - 1][0][2]
                y0 = 0
                y1 = h - 1
                submask = mask[:, x0:x1]
            else:
                y0 = segs[i][0][1]
                y1 = segs[j - 1][0][3]
                x0 = 0
                x1 = w - 1
                submask = mask[y0:y1, :]
            cx_local, cy_local = _intensity_centroid(submask)
            val_candidate = None
            if _is_all_digits_or_sign(txt):
                if axis == "x":
                    frac0, frac1 = (x0 / float(w), x1 / float(w))
                else:
                    frac0, frac1 = (y0 / float(h), y1 / float(h))
                s = txt.replace(" ", "")
                a = max(0, int(round(frac0 * len(s))))
                b = min(len(s), int(round(frac1 * len(s))))
                subs = s[a:b]
                subs = subs.replace(",", ".")
                try:
                    if subs and subs.strip("-").replace(".", "", 1).isdigit():
                        val_candidate = float(subs)
                except Exception:
                    val_candidate = None
            if val_candidate is not None:
                match = _closest_in_llm(val_candidate, llm_vals, atol_abs, atol_rel)
                if match is not None:
                    accepted_j = j
                    accepted_val = match
                    if axis == "x":
                        gx = ox + x0 + cx_local
                        gy = oy + cy_local
                    else:
                        gx = ox + cx_local
                        gy = oy + y0 + cy_local
                    accepted_centroid = (float(gx), float(gy))
                    if axis == "x":
                        accepted_box_local = (ox + x0, oy + 0, ox + x1, oy + h - 1)
                    else:
                        accepted_box_local = (ox + 0, oy + y0, ox + w - 1, oy + y1)
        if accepted_j is None:
            i += 1
        else:
            new_t = dict(token)
            new_t["value"] = float(accepted_val)
            new_t["text"] = str(accepted_val)
            new_t["pixel"] = accepted_centroid
            new_t["bbox"] = accepted_box_local
            refined.append(new_t)
            i = accepted_j
    return refined if refined else [token]


def _coerce_float_list(x):
    out = []
    for v in x or []:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def _refine_axis_tokens_with_valleys(img, tokens, axis, llm_vals, atol_abs, atol_rel):
    out = []
    for t in tokens:
        out.extend(
            _split_merged_digits_with_llm(
                img, t, axis, llm_vals, atol_abs=atol_abs, atol_rel=atol_rel
            )
        )
    return out


class TickCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _snap_tick_to_axis(
    img,
    token,
    axis,
    x_axis_y,
    y_axis_x,
    search_frac=0.02,
    dist_bias_weight=0.3,
    d_flat=6.0,
    d_scale=3.0,
    dist_power=2.0,
    model=None,
    device=torch.device("cpu"),
    min_confidence=0.5,
):
    """
    Model-only tick snapping.
    Returns None if confidence is too low.
    """
    p = token.get("pixel")
    if not (isinstance(p, (list, tuple)) and len(p) == 2):
        return None
    cx, cy = map(int, p)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def dist_bias(dist):
        if dist <= d_flat:
            return 1.0
        x = (dist - d_flat) / d_scale
        return 1.0 / (1.0 + x**dist_power)

    def _extract_tick_crop(gray, cx, cy, size=32):
        H, W = gray.shape
        half = size // 2
        crop = np.ones((size, size), dtype=np.float32)
        x0, x1 = (cx - half, cx + half)
        y0, y1 = (cy - half, cy + half)
        sx0, sx1 = (max(0, x0), min(W, x1))
        sy0, sy1 = (max(0, y0), min(H, y1))
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = gray[sy0:sy1, sx0:sx1]
        return crop

    @torch.no_grad()
    def _model_confidence(crop):
        x = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(device)
        logits = model(x)
        return torch.softmax(logits, dim=1)[0, 1].item()

    if axis == "y":
        axis_x = int(y_axis_x)
        win = max(1, int(H * search_frac))
        y0, y1 = (max(0, cy - win), min(H - 1, cy + win))
        best_y = cy
        best_score = -1000000000.0
        best_conf = 0.0
        for y in range(y0, y1 + 1):
            crop = _extract_tick_crop(gray, axis_x, y)
            conf = _model_confidence(crop)
            dist = abs(y - cy)
            score = conf + dist_bias_weight * dist_bias(dist)
            if score > best_score:
                best_score = score
                best_y = y
                best_conf = conf
        if best_conf < min_confidence:
            return None
        token["pixel"] = (float(axis_x), float(best_y))
        token["snap_confidence"] = float(best_conf)
        return token
    if axis == "x":
        axis_y = int(x_axis_y)
        win = max(1, int(W * search_frac))
        x0, x1 = (max(0, cx - win), min(W - 1, cx + win))
        best_x = cx
        best_score = -1000000000.0
        best_conf = 0.0
        for x in range(x0, x1 + 1):
            crop = _extract_tick_crop(gray, x, axis_y)
            conf = _model_confidence(crop)
            dist = abs(x - cx)
            score = conf + dist_bias_weight * dist_bias(dist)
            if score > best_score:
                best_score = score
                best_x = x
                best_conf = conf
        if best_conf < min_confidence:
            return None
        token["pixel"] = (float(best_x), float(axis_y))
        token["snap_confidence"] = float(best_conf)
        return token
    return None


_NUM_RE = re.compile("^[+-]?\\d+(?:\\.\\d+)?(?:e[+-]?\\d+)?$", re.I)


def _is_numeric_str(s):
    s = str(s).strip().replace("–", "-").replace("—", "-")
    return bool(_NUM_RE.match(s))


def _to_fit_value(v, mode, base=10.0):
    """
    Convert display-domain value to fit-domain value.

    mode:
      - "linear"
      - "log"  (generic log-k, base provided)
    """
    try:
        v = float(v)
    except Exception:
        return None
    if mode == "linear":
        return v
    if mode.startswith("log"):
        if v <= 0 or base <= 0 or base == 1:
            return None
        return math.log(v, base)
    raise ValueError(f"Unknown axis mode: {mode}")


def _tok_text(t):
    if "text" in t and isinstance(t["text"], str):
        return t["text"]
    if "value" in t:
        return str(t["value"])
    return ""


def _tok_pixel_center(t):
    p = t.get("pixel")
    if isinstance(p, (list, tuple)) and len(p) == 2:
        try:
            return (float(p[0]), float(p[1]))
        except Exception:
            return None
    B = t.get("bbox") or t.get("box") or t.get("rect") or t.get("quad")
    if isinstance(B, (list, tuple)) and len(B) == 4:
        x0, y0, x1, y1 = map(float, B)
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    return None


def _intersect_with_llm(
    tokens,
    llm_vals,
    value_tol_abs,
    value_tol_rel,
    axis_span=None,
    pixel_span=None,
    axis="x",
    pixel_tol=0.1,
    breaking=False,
    break_expand=1.3,
    border_factor=1.8,
    mode="linear",
    log_base=10.0,
    H=500,
    W=500,
):
    """
    LLM ∩ OCR gating.

    - Membership is checked ONLY in DISPLAY space (never transformed).
    - Geometry is checked in FIT space using crude LLM span -> pixel span mapping.
    - Supports logK via mode like "log2"/"log10"/"log" + log_base.
    - Border ticks get extra tolerance.
    - Robust to duplicates: for each tick value, keeps only the best residual token.
    - Optional break-aware three-model logic.
    """
    explanations = []
    kept = []
    pixel_tol_px = pixel_tol * (W if axis == "x" else H)
    if not tokens or not llm_vals:
        return ([], [{"selected": False, "reason": "No tokens or no LLM tick values"}])
    llm_vals_arr = np.asarray(llm_vals, float)
    vmin = float(np.min(llm_vals_arr))
    vmax = float(np.max(llm_vals_arr))
    have_span = False
    scale = None
    px0 = px1 = None
    fv0 = fv1 = None
    if axis_span is not None and pixel_span is not None:
        v0, v1 = axis_span
        px0, px1 = pixel_span
        fv0 = _to_fit_value(v0, mode, log_base)
        fv1 = _to_fit_value(v1, mode, log_base)
        if (
            fv0 is not None
            and fv1 is not None
            and np.isfinite(fv0)
            and np.isfinite(fv1)
            and (fv1 != fv0)
            and (px0 is not None)
            and (px1 is not None)
        ):
            scale = (px1 - px0) / float(fv1 - fv0)
            have_span = True
    cand = []
    for t in tokens:
        txt = _tok_text(t)
        p = _tok_pixel_center(t)
        entry = {
            "token": t,
            "text": txt,
            "pixel": p,
            "value": None,
            "selected": False,
            "reason": "",
        }
        if not _is_numeric_str(txt):
            entry["reason"] = f"Rejected: non-numeric text '{txt}'"
            explanations.append(entry)
            continue
        try:
            v = float(txt)
        except Exception:
            entry["reason"] = f"Rejected: cannot parse '{txt}'"
            explanations.append(entry)
            continue
        entry["value"] = v
        atol = max(value_tol_abs, value_tol_rel * max(1.0, abs(v)))
        if not np.any(np.isclose(v, llm_vals_arr, rtol=0.0, atol=atol)):
            entry["reason"] = "Rejected: not in LLM ticks"
            explanations.append(entry)
            continue
        if not have_span or p is None:
            entry["selected"] = True
            entry["reason"] = "Selected: numeric match; no usable span/pixel"
            cand.append(entry)
            continue
        fv = _to_fit_value(v, mode, log_base)
        if fv is None or not np.isfinite(fv):
            entry["reason"] = "Rejected: invalid fit value for axis mode"
            explanations.append(entry)
            continue
        x_px, y_px = p
        px_actual = float(x_px if axis == "x" else y_px)
        if axis == "x":
            px_pred = float(px0 + (fv - fv0) * scale)
        else:
            px_pred = float(px1 - (fv - fv0) * scale)
        if not np.isfinite(px_pred):
            entry["reason"] = "Rejected: non-finite predicted pixel"
            explanations.append(entry)
            continue
        entry["px_actual"] = px_actual
        entry["px_pred"] = px_pred
        entry["residual"] = abs(px_actual - px_pred)
        cand.append(entry)
    if not cand:
        return ([], explanations)
    by_value_best = {}
    for entry in cand:
        v = entry["value"]
        if "residual" in entry:
            tol = pixel_tol_px * (border_factor if v == vmin or v == vmax else 1.0)
            ok = entry["residual"] <= tol
            if not ok and (not breaking):
                entry["selected"] = False
                entry["reason"] = (
                    f"Rejected: residual {entry['residual']:.1f} > {tol:.1f}"
                )
                explanations.append(entry)
                continue
            if not breaking:
                entry["selected"] = True
                entry["reason"] = (
                    f"Selected: residual {entry['residual']:.1f} ≤ {tol:.1f}"
                )
        else:
            entry["selected"] = True
            entry["reason"] = "Selected: membership only (no geometry)"
        prev = by_value_best.get(v)
        if prev is None:
            by_value_best[v] = entry
        else:
            r_new = entry.get("residual", float("inf"))
            r_old = prev.get("residual", float("inf"))
            if r_new < r_old:
                by_value_best[v] = entry
            elif r_new == r_old:
                sc_new = float(entry["token"].get("snap_confidence", 0.0))
                sc_old = float(prev["token"].get("snap_confidence", 0.0))
                if sc_new > sc_old:
                    by_value_best[v] = entry
        explanations.append(entry)
    if breaking and have_span:
        best_entries = list(by_value_best.values())
        kept = []
        axis_range_px = abs(px1 - px0)
        shift = 0.2 * axis_range_px
        comp_factor = 0.8
        scale_C = scale * comp_factor
        tol_A = pixel_tol_px * 1.5 * break_expand
        tol_B = pixel_tol_px * 1.5 * break_expand
        tol_C = pixel_tol_px * 2.0 * break_expand
        for entry in best_entries:
            v = entry["value"]
            p = entry["pixel"]
            if p is None:
                continue
            fv = _to_fit_value(v, mode, log_base)
            if fv is None or not np.isfinite(fv):
                continue
            x_px, y_px = p
            pa = float(x_px if axis == "x" else y_px)
            if axis == "x":
                pxA = float(px0 + (fv - fv0) * scale)
                pxB = float(px0 + shift + (fv - fv0) * scale)
                pxC = float(px0 + (fv - fv0) * scale_C)
            else:
                pxA = float(px1 - (fv - fv0) * scale)
                pxB = float(px1 - shift - (fv - fv0) * scale)
                pxC = float(px1 - (fv - fv0) * scale_C)
            rA = abs(pa - pxA)
            rB = abs(pa - pxB)
            rC = abs(pa - pxC)
            border_mul = border_factor if v == vmin or v == vmax else 1.0
            okA = rA <= tol_A * border_mul
            okB = rB <= tol_B * border_mul
            okC = rC <= tol_C * border_mul
            if okA or okB or okC:
                kept.append(entry["token"])
        return (kept, explanations)
    kept = [e["token"] for e in by_value_best.values() if e.get("selected")]
    return (kept, explanations)


def _fit_axis_from_ticks(tokens, want_axis, mode, log_base=10.0):
    vals_fit = []
    pixs = []

    for t in tokens:
        txt = _tok_text(t)
        if not _is_numeric_str(txt):
            continue

        p = _tok_pixel_center(t)
        if p is None:
            continue

        v_fit = _to_fit_value(float(txt), mode, log_base)
        if v_fit is None:
            continue

        vals_fit.append(v_fit)
        pixs.append(p[0] if want_axis == "x" else p[1])

    if len(vals_fit) < 2:
        return None

    vals_fit = np.asarray(vals_fit, float)
    pixs = np.asarray(pixs, float)
    a, b = majority_polyfit(vals_fit, pixs)

    if want_axis == "x" and a < 0:
        a = abs(a)
    if want_axis == "y" and a > 0:
        a = -abs(a)

    return AxisCal(float(a), float(b), want_axis, mode)


def get_calibration(
    image_path,
    near_tol=10,
    far_tol=5,
    llm_axis_info=None,
    llm_value_tol_abs=1e-06,
    llm_value_tol_rel=0.02,
    res=None,
):
    """
    Step 1:
    check if there was errors in the ocr part (compare it with the llm outputs)
    Step 2:
    put the vertical ticks on the correct x and vice versa
    find exactly where the tick is using Deep Learning snapping tool
    Step 3:
    remove from the ocr results the ticks that are not in the LLM results
    check the consistency in the order of the ticks (using only the bounds of the list of ticks, the point to test and the span)

    Returns:
    tuple: (x_axis, y_axis) containing AxisCal objects for x and y axes.
    """
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_path)
    else:
        img = image_path.copy()
    H, W = img.shape[:2]

    x_axis_y, y_axis_x, x_axis_span, y_axis_span = _detect_axes(img)

    xi = llm_axis_info.get("x_axis", {}) if llm_axis_info else {}
    yi = llm_axis_info.get("y_axis", {}) if llm_axis_info else {}
    x_mode = str(xi.get("type", "linear")).lower() if xi else "linear"
    y_mode = str(yi.get("type", "linear")).lower() if yi else "linear"
    x_break = xi.get("break", False)
    y_break = yi.get("break", False)
    if not x_mode.startswith("log"):
        x_mode = "linear"
    if not y_mode.startswith("log"):
        y_mode = "linear"

    x_llm_ticks = _coerce_float_list(xi.get("ticks"))
    y_llm_ticks = _coerce_float_list(yi.get("ticks"))
    x_limits_llm = (
        tuple(xi.get("range", ()))
        if isinstance(xi.get("range", ()), (list, tuple))
        else None
    )
    y_limits_llm = (
        tuple(yi.get("range", ()))
        if isinstance(yi.get("range", ()), (list, tuple))
        else None
    )

    tokens_all = []
    for plot in (res or {}).get("plots", []):
        tokens_all.extend(plot.get("ticks", []))
        tokens_all.extend(plot.get("labels", []))

    if not x_limits_llm and x_llm_ticks:
        lo, hi = float(min(x_llm_ticks)), float(max(x_llm_ticks))
        pad = 0.05 * max(1.0, hi - lo)
        x_limits_llm = (lo - pad, hi + pad)

    if not y_limits_llm and y_llm_ticks:
        lo, hi = float(min(y_llm_ticks)), float(max(y_llm_ticks))
        pad = 0.05 * max(1.0, hi - lo)
        y_limits_llm = (lo - pad, hi + pad)

    def gather_axis_tokens(tokens, axis, axis_px, other_axis_px):
        near = near_tol * (H / 100 if axis == "x" else W / 100)
        far = far_tol * (W / 100 if axis == "x" else H / 100)
        selected = []

        for t in tokens:
            p = t.get("pixel")
            txt = str(t.get("text"))
            if p is None:
                continue
            try:
                float(txt)
            except Exception:
                continue

            x, y = p
            d_axis = abs((y if axis == "x" else x) - axis_px)
            d_other = abs((x if axis == "x" else y) - other_axis_px)
            if d_axis <= near and d_other > far:
                selected.append(t)

        return selected

    raw_x = gather_axis_tokens(tokens_all, "x", x_axis_y, y_axis_x)
    raw_y = gather_axis_tokens(tokens_all, "y", y_axis_x, x_axis_y)

    # Step 1
    raw_x = _refine_axis_tokens_with_valleys(
        img, raw_x, "x", x_llm_ticks or [], llm_value_tol_abs, llm_value_tol_rel
    )
    raw_y = _refine_axis_tokens_with_valleys(
        img, raw_y, "y", y_llm_ticks or [], llm_value_tol_abs, llm_value_tol_rel
    )
    raw_x = [t for t in raw_x if t.get("axis") == "x"]
    raw_y = [t for t in raw_y if t.get("axis") == "y"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tick_model = TickCNN()
    tick_model_path = Path(__file__).with_name("tick_model.pth")
    tick_model.load_state_dict(torch.load(tick_model_path, map_location=device))
    tick_model.to(device)
    tick_model.eval()

    # Step 2
    raw_x = [
        t
        for t in (
            _snap_tick_to_axis(
                img, tok, "x", x_axis_y, y_axis_x, model=tick_model, device=device
            )
            for tok in raw_x
        )
        if t is not None
    ]
    raw_y = [
        t
        for t in (
            _snap_tick_to_axis(
                img, tok, "y", x_axis_y, y_axis_x, model=tick_model, device=device
            )
            for tok in raw_y
        )
        if t is not None
    ]

    log_base_x = float(x_mode[3:]) if x_mode.startswith("log") else 10
    log_base_y = float(y_mode[3:]) if y_mode.startswith("log") else 10

    # Step 3
    if x_llm_ticks and x_limits_llm:
        raw_x, _ = _intersect_with_llm(
            raw_x,
            x_llm_ticks,
            llm_value_tol_abs,
            llm_value_tol_rel,
            axis_span=x_limits_llm,
            pixel_span=x_axis_span,
            axis="x",
            breaking=x_break,
            H=H,
            W=W,
            mode=x_mode,
            log_base=log_base_x,
        )

    if y_llm_ticks and y_limits_llm:
        raw_y, _ = _intersect_with_llm(
            raw_y,
            y_llm_ticks,
            llm_value_tol_abs,
            llm_value_tol_rel,
            axis_span=y_limits_llm,
            pixel_span=y_axis_span,
            axis="y",
            breaking=y_break,
            H=H,
            W=W,
            mode=y_mode,
            log_base=log_base_y,
        )

    cal_y = _fit_axis_from_ticks(raw_y, "y", y_mode, log_base_y)
    cal_x = _fit_axis_from_ticks(raw_x, "x", x_mode, log_base_x)
    return cal_x, cal_y


# ============================================================================
# Boilerplate helpers (plotting, data shaping, fetch/I-O)
# ============================================================================
# This section intentionally contains only support routines used by the
# notebook orchestration layer. It does not alter calibration behavior.


def draw_calibrated_axes_from_axis_info(ax, x_cal, y_cal, axis_info, W, H):
    """
    Draw calibrated axes ticks on an existing matplotlib axis.
    Assumes image coordinate system (origin top-left).
    """
    x_ticks = axis_info["x_axis"].get("ticks", [])
    for v in x_ticks:
        px = x_cal.v2p(v)
        ax.plot([px, px], [H - 6, H], color="black", lw=1)
        ax.text(px, H - 8, f"{v:g}", ha="center", va="top", fontsize=8)
    y_ticks = axis_info["y_axis"].get("ticks", [])
    for v in y_ticks:
        py = y_cal.v2p(v)
        ax.plot([0, 6], [py, py], color="black", lw=1)
        ax.text(8, py, f"{v:g}", ha="left", va="center", fontsize=8)


# --- Data shaping ------------------------------------------------------------
def normalize_result_to_series(rows):
    """
    Convert flat list of rows into {'series': [{'label', 'points': [...]}, ...]}.
    """
    grouped = defaultdict(list)
    for r in rows:
        sid = r["series_id"]
        grouped[sid].append({"x": r["x"], "y": r["y"]})
    series = []
    for i, (sid, pts) in enumerate(grouped.items()):
        series.append({"label": sid, "points": pts})
    return {"series": series}


# --- Notebook display --------------------------------------------------------
def show_extracted_series_overlay(result, image_path, x_cal, y_cal, axis_info, dpi=150):
    """
    Display a combined overlay in the notebook (no files are written).
    """
    series_list = result.get("series", [])
    if not series_list:
        print("No series to plot.")
        return

    if isinstance(image_path, str):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
    else:
        img_bgr = image_path.copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(img_rgb)

    for i, s in enumerate(series_list):
        pts = s.get("points", [])
        if not pts:
            continue
        x_raw = np.array([float(p["x"]) for p in pts])
        y_raw = np.array([float(p["y"]) for p in pts])
        ax.plot(
            x_cal.v2p(x_raw),
            y_cal.v2p(y_raw),
            lw=2,
            label=s.get("label", f"{i}"),
        )

    draw_calibrated_axes_from_axis_info(ax, x_cal, y_cal, axis_info, W, H)
    ax.legend(fontsize=8)
    ax.axis("off")
    plt.show()
    plt.close(fig)


# --- Image retrieval & misc --------------------------------------------------
def fetch_images_from_source_sid(extract_sid, token, graphql_url):
    headers = {"Cookie": f"{token}", "Content-Type": "application/json"}
    query = """
    query ProjectItemAppPanel($id: ID!) {
      projectItem(id: $id) {
        id: sid
        content {
          ... on Assertion {
            id
            highlights {
              ... on RectangularHighlight {
                id
                picture
              }
            }
          }
        }
      }
    }
    """
    try:
        response = requests.post(
            graphql_url,
            headers=headers,
            json={"query": query, "variables": {"id": extract_sid}},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise ValueError(
            f"Could not contact Jinko GraphQL API at {graphql_url}: {exc}"
        ) from exc

    if response.status_code != 200:
        raise ValueError(
            f"Jinko GraphQL request failed with HTTP {response.status_code}. "
            f"Check TOKEN permissions and graphql_url={graphql_url}."
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("Jinko GraphQL response is not valid JSON.") from exc

    if data.get("errors"):
        msg = "; ".join(str(e.get("message", e)) for e in data["errors"])
        raise ValueError(
            f"Jinko GraphQL returned errors for extract_sid={extract_sid}: {msg}"
        )

    project_item = (data.get("data") or {}).get("projectItem") or {}
    content = project_item.get("content") or {}
    highlights = content.get("highlights") or []

    if not project_item:
        raise ValueError(
            f"No project item found for extract_sid={extract_sid}. "
            "Check EXTRACT_SID and access rights."
        )

    if not highlights:
        raise ValueError(
            f"No highlight images found for extract_sid={extract_sid}. "
            "The assertion may have no picture highlights."
        )

    imgs = []
    image_bytes = []
    image_mimes = []
    errors = []

    for h in highlights:
        img_url = h.get("picture")
        if not img_url:
            errors.append("missing picture URL in highlight entry")
            continue

        try:
            img_resp = requests.get(img_url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            errors.append(f"failed downloading {img_url}: {exc}")
            continue

        if img_resp.status_code != 200:
            errors.append(f"HTTP {img_resp.status_code} when downloading {img_url}")
            continue

        arr = np.frombuffer(img_resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            errors.append(f"could not decode image bytes from {img_url}")
            continue

        imgs.append(img)
        image_bytes.append(img_resp.content)
        mime = (
            (img_resp.headers.get("Content-Type") or "image/png")
            .split(";")[0]
            .strip()
            .lower()
        )
        if not mime.startswith("image/"):
            mime = "image/png"
        image_mimes.append(mime)

    if not imgs:
        extra = f" Details: {errors[0]}" if errors else ""
        raise ValueError(
            f"Found highlights for extract_sid={extract_sid}, "
            f"but none could be downloaded/decoded.{extra}"
        )

    return (imgs[0], image_bytes[0], image_mimes[0])


def fetch_images_from_source_sid_simplified(
    extract_sid, token, graphql_url="https://api.jinko.ai/_api"
):
    headers = {"Cookie": f"{token}", "Content-Type": "application/json"}
    query = """
    query ProjectItemAppPanel($id: ID!) {
      projectItem(id: $id) {
        id: sid
        content {
          ... on Assertion {
            id
            highlights {
              ... on RectangularHighlight {
                id
                picture
              }
            }
          }
        }
      }
    }
    """
    response = requests.post(
        graphql_url,
        headers=headers,
        json={"query": query, "variables": {"id": extract_sid}},
        timeout=30,
    )

    data = response.json()

    project_item = (data.get("data") or {}).get("projectItem") or {}
    content = project_item.get("content") or {}
    highlights = content.get("highlights") or []

    img_urls = []

    for h in highlights:
        img_urls = [h.get("picture") for h in highlights]

    return img_urls[0]
