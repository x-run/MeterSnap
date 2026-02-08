"""
meter_reader.py — 針検出 + 角度→値変換エンジン

公開関数:
  find_gauge_center(image_path)          → (cx, cy, radius)
  detect_needle_angle(image_path, cx, cy, radius, debug_dir=None) → float | None
  angle_to_value(needle_angle, calibration_points, direction="cw") → float | None
"""

import json
import math
import os

import cv2
import numpy as np


# ===========================================================
# 1. ゲージ中心の自動検出
# ===========================================================

def find_gauge_center(image_path):
    """
    HoughCircles でゲージの円を検出し (cx, cy, radius) を返す。
    検出できない場合は画像中心をフォールバックとして返す。
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1.2,
        minDist=min(h, w) // 3,
        param1=100, param2=50,
        minRadius=min(h, w) // 6,
        maxRadius=min(h, w) // 2,
    )

    if circles is not None:
        c = circles[0][np.argmax(circles[0][:, 2])]
        return int(c[0]), int(c[1]), int(c[2])

    return w // 2, h // 2, min(h, w) // 3


# ===========================================================
# 2. 針の角度検出（放射状スキャン方式）
# ===========================================================

def _create_needle_mask(img, cx=None, cy=None, radius=None):
    """赤針専用マスク — HSV赤色のみ抽出 + ROIリング"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色の2レンジ (H が 0 付近と 180 付近に分かれる)
    red1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    mask = red1 | red2

    # morph open → close: 点ノイズ除去 → 針の途切れを繋ぐ
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # ROI リング: 半径 20%〜90% だけ残す
    if cx is not None and cy is not None and radius is not None:
        h, w = img.shape[:2]
        ys, xs = np.ogrid[:h, :w]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        ring = (dist >= radius * 0.20) & (dist <= radius * 0.90)
        mask[~ring] = 0

    return mask


def detect_needle_angle(image_path, cx, cy, radius, debug_dir=None):
    """
    放射状スキャンで針角度を検出する。

    座標系: 右=0°, 反時計回りが正 (数学座標系)
    スキャン範囲: 半径 20%〜90% (中心ハブと外周を除外)

    debug_dir: 指定すると mask.png / overlay.png / meta.json を保存

    Returns:
        float — 角度 (度, 丸めなし)。検出失敗時は None。
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    mask = _create_needle_mask(img, cx, cy, radius)

    r_start, r_end = 0.20, 0.90
    n_steps = max(5, int(radius * (r_end - r_start)))

    scores = np.zeros(360, dtype=np.float64)
    for deg in range(360):
        rad = math.radians(deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cnt = 0
        for s in range(n_steps):
            r = radius * (r_start + (r_end - r_start) * s / max(1, n_steps - 1))
            x, y = int(cx + r * cos_a), int(cy - r * sin_a)
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                cnt += 1
        scores[deg] = cnt

    if scores.max() < 3:
        _save_debug(debug_dir, img, mask, cx, cy, radius, None, scores)
        return None

    # 5° 幅の移動平均でスムージング
    sm = np.array([np.mean([scores[(d + k) % 360] for k in range(-2, 3)])
                   for d in range(360)])

    # ピーク → クラスタリング
    thr = sm.max() * 0.6
    clusters, cur = [], []
    for d in range(360):
        if sm[d] >= thr:
            cur.append(d)
        elif cur:
            clusters.append(cur)
            cur = []
    if cur:
        clusters.append(cur)
    if len(clusters) > 1 and clusters[-1][-1] >= 358 and clusters[0][0] <= 1:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()
    if not clusters:
        _save_debug(debug_dir, img, mask, cx, cy, radius, None, scores)
        return None

    # 各クラスタの加重平均角度を算出
    best = max(clusters, key=lambda c: sum(sm[a % 360] for a in c))
    s_sin = sum(math.sin(math.radians(a)) * sm[a % 360] for a in best)
    s_cos = sum(math.cos(math.radians(a)) * sm[a % 360] for a in best)
    angle = math.degrees(math.atan2(s_sin, s_cos))
    if angle < 0:
        angle += 360

    _save_debug(debug_dir, img, mask, cx, cy, radius, angle, scores)
    return angle


def _save_debug(debug_dir, img, mask, cx, cy, radius, angle, scores):
    """debug_dir が指定されていれば mask.png / overlay.png / meta.json を保存"""
    if debug_dir is None:
        return
    os.makedirs(debug_dir, exist_ok=True)

    # mask.png — ROI適用済み赤針マスク
    cv2.imwrite(os.path.join(debug_dir, "mask.png"), mask)

    # overlay.png — 元画像に中心・ROIリング・針線を描画
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)                # 中心点
    cv2.circle(overlay, (cx, cy), int(radius * 0.20), (255, 255, 0), 1)  # 内側境界
    cv2.circle(overlay, (cx, cy), int(radius * 0.90), (255, 255, 0), 1)  # 外側境界

    # 上位5クラスタの候補角度を灰色で描画
    sm = np.array([np.mean([scores[(d + k) % 360] for k in range(-2, 3)])
                   for d in range(360)])
    thr = sm.max() * 0.6 if sm.max() > 0 else 0
    for d in range(360):
        if sm[d] >= thr:
            rad = math.radians(d)
            ex = int(cx + radius * 0.90 * math.cos(rad))
            ey = int(cy - radius * 0.90 * math.sin(rad))
            cv2.line(overlay, (cx, cy), (ex, ey), (128, 128, 128), 1)

    # 最終採用角度を赤太線で描画
    if angle is not None:
        rad = math.radians(angle)
        ex = int(cx + radius * 0.95 * math.cos(rad))
        ey = int(cy - radius * 0.95 * math.sin(rad))
        cv2.line(overlay, (cx, cy), (ex, ey), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(debug_dir, "overlay.png"), overlay)

    # meta.json
    top5 = sorted(range(360), key=lambda d: -sm[d])[:5]
    meta = {
        "cx": cx, "cy": cy, "radius": radius,
        "needle_angle": angle,
        "score_max": float(scores.max()),
        "top5_degrees": [{"deg": d, "score": float(sm[d])} for d in top5],
    }
    with open(os.path.join(debug_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ===========================================================
# 3. 角度 → 値 変換（区間線形補間）
# ===========================================================

def angle_to_value(needle_angle, calibration_points, direction="cw"):
    """
    キャリブレーション点に基づく区間線形補間 (piecewise linear)。

    calibration_points: [{"angle": float, "value": float}, ...]
      - value 昇順にソート済みであること
      - 最低 2 点必要

    direction: "cw" | "ccw"
      - cw  (時計回り): 正規化 = (start − angle) mod 360
      - ccw (反時計回り): 正規化 = (angle − start) mod 360

    Returns:
        float — 補間値 (丸めなし)。範囲外の場合は None。
    """
    pts = sorted(calibration_points, key=lambda p: p["value"])
    if len(pts) < 2:
        return None

    start = pts[0]["angle"]

    if direction == "ccw":
        norms = [(p["angle"] - start) % 360 for p in pts]
    else:
        norms = [(start - p["angle"]) % 360 for p in pts]
    vals = [p["value"] for p in pts]

    # 単調性を保証 (同一角度対策)
    for i in range(1, len(norms)):
        if norms[i] <= norms[i - 1]:
            norms[i] = norms[i - 1] + 0.01

    if direction == "ccw":
        n_needle = (needle_angle - start) % 360
    else:
        n_needle = (start - needle_angle) % 360

    # 範囲外チェック (5° のマージンを許容)
    margin = 5.0
    if n_needle < norms[0] - margin or n_needle > norms[-1] + margin:
        return None

    n_needle = max(norms[0], min(norms[-1], n_needle))
    return float(np.interp(n_needle, norms, vals))
