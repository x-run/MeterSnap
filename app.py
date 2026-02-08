"""
MeterSnap v3 — メーター読み取りWebアプリ
テンプレート登録（中心クリック + ドラッグ式キャリブレーション）
→ 針角度検出 → 区間線形補間で値変換 → CSV追記
"""

import os, json, csv, uuid
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify,
    send_file, send_from_directory,
)
from meter_reader import find_gauge_center, detect_needle_angle, angle_to_value

app = Flask(__name__)

# ===== パス定数 =====
BASE      = os.path.dirname(__file__)
UPLOADS   = os.path.join(BASE, "data", "uploads")
METERS    = os.path.join(BASE, "data", "meters")
CSV_FILE  = os.path.join(BASE, "data", "readings.csv")
EXTS      = {"png", "jpg", "jpeg", "bmp", "webp"}

for d in (UPLOADS, METERS):
    os.makedirs(d, exist_ok=True)


# ===== ヘルパー =====

def _allowed(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in EXTS

def _save_upload(file):
    ext = file.filename.rsplit(".", 1)[1].lower()
    name = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6] + "." + ext
    path = os.path.join(UPLOADS, name)
    file.save(path)
    return name, path

def _load_meter(meter_id):
    p = os.path.join(METERS, f"{meter_id}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def _save_meter(data):
    p = os.path.join(METERS, f"{data['id']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _list_meters():
    out = []
    for fn in sorted(os.listdir(METERS)):
        if fn.endswith(".json"):
            with open(os.path.join(METERS, fn), encoding="utf-8") as f:
                out.append(json.load(f))
    return out

def _append_csv(meter_id, value, unit, image_filename):
    exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "meter_id", "value", "unit", "image_filename"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            meter_id, value, unit, image_filename,
        ])

def _recent_readings(limit=30):
    if not os.path.exists(CSV_FILE):
        return []
    with open(CSV_FILE, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    return rows[-limit:][::-1] if len(rows) > 1 else []


# ===== ページ =====

@app.route("/")
def index():
    return render_template("index.html",
                           meters=_list_meters(),
                           readings=_recent_readings())

@app.route("/meters/new")
def meter_new():
    return render_template("meter_new.html")

@app.route("/meters/<meter_id>")
def meter_detail(meter_id):
    m = _load_meter(meter_id)
    if not m:
        return "not found", 404
    return render_template("meter_new.html", meter=m)

@app.route("/read")
def read_page():
    return render_template("read.html", meters=_list_meters())


# ===== API: メーターテンプレート =====

@app.route("/api/meters", methods=["POST"])
def api_save_meter():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify(success=False, error="名前を入力してください")
    pts = data.get("calibration_points", [])
    if len(pts) < 2:
        return jsonify(success=False, error="キャリブレーション点を2つ以上登録してください")

    meter_id = data.get("id") or uuid.uuid4().hex[:8]
    direction = data.get("direction", "cw")
    if direction not in ("cw", "ccw"):
        direction = "cw"
    meter = {
        "id": meter_id,
        "name": data["name"],
        "unit": data.get("unit", ""),
        "image": data.get("image", ""),
        "image_width": data.get("image_width", 0),
        "image_height": data.get("image_height", 0),
        "center": data.get("center", {"x": 0, "y": 0}),
        "direction": direction,
        "calibration_points": sorted(pts, key=lambda p: p["value"]),
        "created": data.get("created") or datetime.now().isoformat(),
    }
    _save_meter(meter)
    return jsonify(success=True, meter=meter)

@app.route("/api/meters/<meter_id>", methods=["DELETE"])
def api_delete_meter(meter_id):
    p = os.path.join(METERS, f"{meter_id}.json")
    if os.path.exists(p):
        os.remove(p)
    return jsonify(success=True)


# ===== API: 画像関連 =====

@app.route("/api/upload-image", methods=["POST"])
def api_upload_image():
    """画像アップロード → ゲージ中心自動検出"""
    f = request.files.get("image")
    if not f or not _allowed(f.filename):
        return jsonify(success=False, error="画像を選択してください")

    name, path = _save_upload(f)
    cx, cy, radius = find_gauge_center(path)

    import cv2
    img = cv2.imread(path)
    h, w = img.shape[:2]

    return jsonify(success=True,
                   filename=name,
                   image_width=w, image_height=h,
                   cx=cx, cy=cy, radius=radius)

@app.route("/data/uploads/<path:fn>")
def serve_upload(fn):
    return send_from_directory(UPLOADS, fn)

@app.route("/data/debug/<path:fn>")
def serve_debug(fn):
    return send_from_directory(os.path.join(BASE, "data", "debug"), fn)


# ===== API: 読み取り =====

@app.route("/api/read", methods=["POST"])
def api_read():
    f = request.files.get("image")
    meter_id = request.form.get("meter_id")

    if not f or not _allowed(f.filename):
        return jsonify(success=False, error="画像を選択してください")
    if not meter_id:
        return jsonify(success=False, error="メーターを選択してください")

    meter = _load_meter(meter_id)
    if not meter:
        return jsonify(success=False, error="テンプレートが見つかりません")

    img_name, img_path = _save_upload(f)

    # 読み取り画像で円を再検出
    cx, cy, radius = find_gauge_center(img_path)
    if cx is None:
        return jsonify(success=False,
                       error="メーターを検出できませんでした。撮り直してください。")

    # 針角度を検出（デバッグ画像を毎回保存）
    debug_dir = os.path.join(BASE, "data", "debug", img_name.rsplit(".", 1)[0])
    angle = detect_needle_angle(img_path, cx, cy, radius, debug_dir=debug_dir)
    if angle is None:
        return jsonify(success=False,
                       error="針を検出できませんでした。撮り直してください。")

    # 角度→値 変換（direction 対応）
    direction = meter.get("direction", "cw")
    value = angle_to_value(angle, meter["calibration_points"], direction=direction)
    if value is None:
        return jsonify(success=False,
                       error="針が目盛り範囲外です。撮り直してください。")

    # 表示用に丸め
    value_display = round(value, 1)

    # CSV 追記
    _append_csv(meter_id, value_display, meter.get("unit", ""), img_name)

    return jsonify(success=True,
                   value=value_display,
                   value_raw=value,
                   unit=meter.get("unit", ""),
                   angle=round(angle, 1),
                   angle_raw=angle,
                   meter_name=meter["name"],
                   debug_dir=debug_dir)


# ===== CSV ダウンロード =====

@app.route("/download")
def download():
    if not os.path.exists(CSV_FILE):
        return "データなし", 404
    return send_file(CSV_FILE, as_attachment=True,
                     download_name="meter_readings.csv")


# ===== 起動 =====

if __name__ == "__main__":
    print("=" * 50)
    print("  MeterSnap v3")
    print("  http://localhost:5001")
    print("  Ctrl+C で停止")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=True)
