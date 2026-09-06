import base64
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "billboard_best.pt"
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUT_DIR = STATIC_DIR / "outputs"

# ---------------------------------------------------------------------------
# YOLO config dir — Render's home dir is read-only, force /tmp
# ---------------------------------------------------------------------------
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# ---------------------------------------------------------------------------
# Global model handle (lazy-loaded)
# ---------------------------------------------------------------------------
_model = None


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def download_model_if_needed() -> None:
    """Download model weights from MODEL_URL env var if not present on disk."""
    if MODEL_PATH.exists():
        return

    model_url = os.environ.get("MODEL_URL")
    if not model_url:
        print("No MODEL_URL set and no model file found — skipping download.")
        return

    print(f"Downloading model weights from MODEL_URL → {MODEL_PATH} …")

    import requests

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(model_url, stream=True, timeout=300)
    resp.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    print("Model download complete ✓")


def get_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained model found at {MODEL_PATH}. "
                "Upload models/billboard_best.pt or set MODEL_URL in Render env vars."
            ),
        )

    from ultralytics import YOLO

    print(f"Loading YOLO model from {MODEL_PATH} …")

    try:
        _model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"Standard load failed ({e}); retrying with safe_globals …")

        import torch

        try:
            from ultralytics.nn.tasks import DetectionModel

            torch.serialization.add_safe_globals([DetectionModel])
        except ImportError:
            pass

        _model = YOLO(str(MODEL_PATH))

    print("Model loaded ✓")
    return _model


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    download_model_if_needed()

    if MODEL_PATH.exists():
        try:
            get_model()
        except Exception as e:
            print(f"Warning: model failed to load at startup: {e}")
    else:
        print(
            "Warning: model file not found. "
            "Set MODEL_URL env var or commit models/billboard_best.pt."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Billboard Ad Replacement", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def read_upload_as_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    return img


def bgr_to_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)

    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image.")

    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Detection / quad-fitting / compositing
# ---------------------------------------------------------------------------
def detect_billboards(model, image_bgr: np.ndarray, conf_threshold: float):
    results = model.predict(image_bgr, conf=conf_threshold, verbose=False)
    boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))

    return boxes


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered


def box_to_quad(box, image_shape) -> np.ndarray:
    x1, y1, x2, y2, _ = box
    h, w = image_shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def quad_to_payload(quad: np.ndarray):
    return [{"x": float(x), "y": float(y)} for x, y in quad.reshape(4, 2)]


def parse_quad_payload(quad_json: Optional[str]) -> Optional[np.ndarray]:
    if not quad_json:
        return None

    try:
        raw = json.loads(quad_json)
        pts = np.array([[p["x"], p["y"]] for p in raw], dtype=np.float32)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None

    if pts.shape != (4, 2):
        return None

    return order_points(pts)


def is_valid_quad(quad: np.ndarray, image_shape, min_area: float = 200.0) -> bool:
    h, w = image_shape[:2]

    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return False

    if (quad[:, 0] < -2).any() or (quad[:, 0] > w + 2).any():
        return False

    if (quad[:, 1] < -2).any() or (quad[:, 1] > h + 2).any():
        return False

    if not cv2.isContourConvex(quad.astype(np.float32)):
        return False

    area = abs(cv2.contourArea(quad.astype(np.float32)))
    if area < min_area:
        return False

    widths = [
        np.linalg.norm(quad[1] - quad[0]),
        np.linalg.norm(quad[2] - quad[3]),
    ]
    heights = [
        np.linalg.norm(quad[3] - quad[0]),
        np.linalg.norm(quad[2] - quad[1]),
    ]

    return min(widths + heights) >= 8


def quad_matches_box(
    quad: np.ndarray,
    box,
    min_width_ratio: float = 0.78,
    min_height_ratio: float = 0.70,
) -> bool:
    x1, y1, x2, y2, _ = box

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    qx, qy, qw, qh = cv2.boundingRect(quad.astype(np.float32))

    width_ratio = qw / box_w
    height_ratio = qh / box_h

    box_cx = (x1 + x2) / 2
    box_cy = (y1 + y2) / 2
    quad_cx = qx + qw / 2
    quad_cy = qy + qh / 2

    center_dx = abs(quad_cx - box_cx) / box_w
    center_dy = abs(quad_cy - box_cy) / box_h

    return (
        width_ratio >= min_width_ratio
        and height_ratio >= min_height_ratio
        and center_dx <= 0.22
        and center_dy <= 0.22
    )


def line_from_segment(seg):
    x1, y1, x2, y2 = map(float, seg)

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    norm = (a * a + b * b) ** 0.5
    if norm == 0:
        return None

    return np.array([a / norm, b / norm, c / norm], dtype=np.float32)


def intersect_lines(line_a, line_b):
    p = np.cross(line_a, line_b)

    if abs(p[2]) < 1e-5:
        return None

    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float32)


def quad_from_hough_lines(crop: np.ndarray, box_in_crop) -> Optional[np.ndarray]:
    bx1, by1, bx2, by2 = box_in_crop

    box_w = max(1, bx2 - bx1)
    box_h = max(1, by2 - by1)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    median = float(np.median(gray))
    lower = int(max(20, 0.66 * median))
    upper = int(min(180, 1.33 * median + 30))

    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    min_len = int(max(35, min(box_w, box_h) * 0.35))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=min_len,
        maxLineGap=18,
    )

    if lines is None:
        return None

    horizontal = []
    vertical = []

    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, line)
        dx = x2 - x1
        dy = y2 - y1

        length = (dx * dx + dy * dy) ** 0.5
        if length < min_len:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))
        angle = min(angle, 180 - angle)

        model_line = line_from_segment(line)
        if model_line is None:
            continue

        if angle <= 25:
            center_x = (bx1 + bx2) / 2

            if abs(dx) < 1:
                continue

            y_at_center = y1 + (dy / dx) * (center_x - x1)
            horizontal.append((model_line, y_at_center, length))

        elif angle >= 65:
            center_y = (by1 + by2) / 2

            if abs(dy) < 1:
                continue

            x_at_center = x1 + (dx / dy) * (center_y - y1)
            vertical.append((model_line, x_at_center, length))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    def best_boundary(candidates, target_pos):
        return min(
            candidates,
            key=lambda item: abs(item[1] - target_pos) - 0.03 * item[2],
        )[0]

    top = best_boundary(horizontal, by1)
    bottom = best_boundary(horizontal, by2)
    left = best_boundary(vertical, bx1)
    right = best_boundary(vertical, bx2)

    corners = [
        intersect_lines(top, left),
        intersect_lines(top, right),
        intersect_lines(bottom, right),
        intersect_lines(bottom, left),
    ]

    if any(p is None for p in corners):
        return None

    return order_points(np.array(corners, dtype=np.float32))


def quad_from_contours(crop: np.ndarray, box_in_crop) -> Optional[np.ndarray]:
    bx1, by1, bx2, by2 = box_in_crop
    expected_area = max(1, (bx2 - bx1) * (by2 - by1))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 0.25 * expected_area or area > 1.45 * expected_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        for epsilon in (0.015, 0.025, 0.04, 0.06):
            approx = cv2.approxPolyDP(cnt, epsilon * perimeter, True)

            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = order_points(approx.astype(np.float32))
            x, y, w, h = cv2.boundingRect(approx)

            center_error = (
                abs((x + w / 2) - (bx1 + bx2) / 2)
                + abs((y + h / 2) - (by1 + by2) / 2)
            )

            score = area - 2.0 * center_error

            if score > best_score:
                best = quad
                best_score = score

            break

    return best


def find_billboard_quad(scene_bgr: np.ndarray, box, padding_ratio: float = 0.08):
    x1, y1, x2, y2, _ = box
    h, w = scene_bgr.shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_y)

    fallback_quad = box_to_quad(box, scene_bgr.shape)

    crop = scene_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return fallback_quad, True

    box_in_crop = (
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1,
    )

    for finder in (quad_from_hough_lines, quad_from_contours):
        quad = finder(crop, box_in_crop)

        if quad is None:
            continue

        quad[:, 0] += crop_x1
        quad[:, 1] += crop_y1

        if (
            is_valid_quad(quad, scene_bgr.shape, min_area=0.2 * box_w * box_h)
            and quad_matches_box(quad, box)
        ):
            return quad, False

    return fallback_quad, True


def warp_and_composite(scene_bgr, ad_bgr, quad, blend_edge: bool = True):
    h_scene, w_scene = scene_bgr.shape[:2]
    h_ad, w_ad = ad_bgr.shape[:2]

    src_pts = np.array(
        [[0, 0], [w_ad, 0], [w_ad, h_ad], [0, h_ad]],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src_pts, quad)

    warped_ad = cv2.warpPerspective(
        ad_bgr,
        H,
        (w_scene, h_scene),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    mask = np.zeros((h_scene, w_scene), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)

    if blend_edge:
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

    composite = (
        warped_ad.astype(np.float32) * mask_3ch
        + scene_bgr.astype(np.float32) * (1 - mask_3ch)
    )

    return composite.astype(np.uint8)


# ---------------------------------------------------------------------------
# API routes — register BEFORE mounting StaticFiles
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_file_present": MODEL_PATH.exists(),
    }


@app.post("/api/detect")
async def api_detect(scene: UploadFile = File(...), conf: float = Form(0.25)):
    model = get_model()

    scene_bytes = await scene.read()
    scene_bgr = read_upload_as_bgr(scene_bytes)

    boxes = detect_billboards(model, scene_bgr, conf)

    detections = []
    preview = scene_bgr.copy()

    for i, b in enumerate(boxes):
        x1, y1, x2, y2, c = b

        quad, used_fallback = find_billboard_quad(scene_bgr, b)

        detections.append(
            {
                "index": i,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": c,
                "quad": quad_to_payload(quad),
                "used_fallback": used_fallback,
            }
        )

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.polylines(
            preview,
            [quad.astype(np.int32)],
            isClosed=True,
            color=(0, 140, 255),
            thickness=3,
        )

        cv2.putText(
            preview,
            f"#{i + 1} {c:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    h, w = scene_bgr.shape[:2]

    return JSONResponse(
        {
            "boxes": detections,
            "preview": bgr_to_data_url(preview),
            "width": w,
            "height": h,
        }
    )


@app.post("/api/composite")
async def api_composite(
    scene: UploadFile = File(...),
    ad: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    fit_shape: bool = Form(True),
    blend_edges: bool = Form(True),
    quad_json: Optional[str] = Form(None),
):
    scene_bytes = await scene.read()
    ad_bytes = await ad.read()

    scene_bgr = read_upload_as_bgr(scene_bytes)
    ad_bgr = read_upload_as_bgr(ad_bytes)

    box = (int(x1), int(y1), int(x2), int(y2), 1.0)

    used_fallback = True

    if fit_shape:
        quad = parse_quad_payload(quad_json)

        if (
            quad is not None
            and is_valid_quad(quad, scene_bgr.shape)
            and quad_matches_box(quad, box)
        ):
            used_fallback = False
        else:
            quad, used_fallback = find_billboard_quad(scene_bgr, box)
    else:
        quad = box_to_quad(box, scene_bgr.shape)

    result_bgr = warp_and_composite(
        scene_bgr,
        ad_bgr,
        quad,
        blend_edge=blend_edges,
    )

    preview = scene_bgr.copy()

    cv2.rectangle(
        preview,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2,
    )

    cv2.polylines(
        preview,
        [quad.astype(np.int32)],
        isClosed=True,
        color=(0, 140, 255),
        thickness=3,
    )

    return JSONResponse(
        {
            "result": bgr_to_data_url(result_bgr),
            "preview": bgr_to_data_url(preview),
            "used_fallback": used_fallback,
            "quad": quad_to_payload(quad),
        }
    )



def tracked_billboards_from_result(result):
    tracked = []

    if result.boxes is None or len(result.boxes) == 0:
        return tracked

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(xyxy))

    if result.boxes.id is None:
        ids = np.arange(len(xyxy), dtype=np.int32)
    else:
        ids = result.boxes.id.cpu().numpy().astype(np.int32)

    for coords, conf, track_id in zip(xyxy, confs, ids):
        x1, y1, x2, y2 = coords
        tracked.append((int(track_id), int(x1), int(y1), int(x2), int(y2), float(conf)))

    return tracked


def choose_track(detections, target_track_id: Optional[int], previous_box=None):
    if not detections:
        return None, target_track_id

    if target_track_id is not None:
        for det in detections:
            if det[0] == target_track_id:
                return det, target_track_id

    if previous_box is not None:
        px1, py1, px2, py2, _ = previous_box
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2

        def distance_to_previous(det):
            _, x1, y1, x2, y2, _ = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            return (cx - pcx) ** 2 + (cy - pcy) ** 2

        chosen = min(detections, key=distance_to_previous)
        return chosen, chosen[0]

    chosen = max(detections, key=lambda det: (det[3] - det[1]) * (det[4] - det[2]))
    return chosen, chosen[0]


def process_video_with_bytetrack(
    video_path: str,
    ad_bgr: np.ndarray,
    output_path: str,
    conf_threshold: float,
    fit_shape: bool,
    blend_edges: bool,
    max_frames: Optional[int] = None,
):
    model = get_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Uploaded video has invalid dimensions.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Could not create output video.")

    target_track_id = None
    previous_box = None
    processed_frames = 0
    composited_frames = 0
    fallback_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_frames is not None and processed_frames >= max_frames:
                break

            results = model.track(
                frame,
                conf=conf_threshold,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )

            detections = []
            for result in results:
                detections.extend(tracked_billboards_from_result(result))

            chosen, target_track_id = choose_track(detections, target_track_id, previous_box)

            if chosen is not None:
                _, x1, y1, x2, y2, conf = chosen
                box = (x1, y1, x2, y2, conf)

                if fit_shape:
                    quad, used_fallback = find_billboard_quad(frame, box)
                else:
                    quad, used_fallback = box_to_quad(box, frame.shape), True

                frame = warp_and_composite(frame, ad_bgr, quad, blend_edge=blend_edges)
                previous_box = box
                composited_frames += 1

                if used_fallback:
                    fallback_frames += 1

            writer.write(frame)
            processed_frames += 1
    finally:
        cap.release()
        writer.release()

    return {
        "frames": processed_frames,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "composited_frames": composited_frames,
        "fallback_frames": fallback_frames,
        "track_id": target_track_id,
    }


@app.post("/api/video/composite")
async def api_video_composite(
    video: UploadFile = File(...),
    ad: UploadFile = File(...),
    conf: float = Form(0.25),
    fit_shape: bool = Form(True),
    blend_edges: bool = Form(True),
    max_frames: Optional[int] = Form(None),
):
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        suffix = ".mp4"

    ad_bgr = read_upload_as_bgr(await ad.read())
    video_bytes = await video.read()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"billboard_video_{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_DIR / output_name

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            temp_path = tmp.name

        stats = process_video_with_bytetrack(
            temp_path,
            ad_bgr,
            str(output_path),
            conf_threshold=conf,
            fit_shape=fit_shape,
            blend_edges=blend_edges,
            max_frames=max_frames,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    return JSONResponse(
        {
            "video_url": f"/static/outputs/{output_name}",
            "download_url": f"/static/outputs/{output_name}",
            **stats,
        }
    )
# ---------------------------------------------------------------------------
# Serve index.html at root — must come BEFORE the static mount
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static/")

    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# Static assets — mount LAST so API routes take priority
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"Warning: static dir not found at {STATIC_DIR}")