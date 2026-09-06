# Space Swap - Billboard Ad Replacement

Space Swap is a FastAPI web app for replacing billboard artwork in images and videos. It uses a custom Ultralytics YOLO model to detect billboards, OpenCV perspective warping to paste an uploaded ad creative onto the detected billboard area, and ByteTrack to keep the selected billboard stable across video frames.

## Features

- Upload a scene image and an ad creative.
- Detect billboard panels with a custom YOLO model.
- Fit the ad to a detected four-corner billboard shape when possible.
- Fall back to a rectangle when clean corners cannot be found.
- Feather composite edges for a softer paste.
- Upload a scene video and process it frame by frame.
- Track the billboard across video frames with ByteTrack.
- Export processed image proofs and MP4 video proofs.

## Project Structure

```text
.
├── app.py
├── models/
│   └── billboard_best.pt
└── static/
    ├── index.html
    ├── script.js
    ├── style.css
    └── outputs/
```

`models/billboard_best.pt` is required unless you set `MODEL_URL` so the app can download the model at startup.

`static/outputs/` is where processed video files are written.

## Requirements

Use Python 3.10+ if possible.

Install the required packages:

```bash
pip install fastapi uvicorn python-multipart opencv-python numpy ultralytics requests
```

If you are running on a server without GUI libraries, use:

```bash
pip install fastapi uvicorn python-multipart opencv-python-headless numpy ultralytics requests
```

## Model Setup

Place your trained billboard detector here:

```text
models/billboard_best.pt
```

Alternatively, set an environment variable:

```bash
MODEL_URL=https://your-model-url/billboard_best.pt
```

If `models/billboard_best.pt` is missing and `MODEL_URL` is not set, the app will start but detection endpoints will return a model error.

## Run Locally

From the project folder:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Image Workflow

1. Upload a scene photo containing a billboard.
2. Upload the ad creative you want to insert.
3. Adjust detection confidence if needed.
4. Click **Run Detection**.
5. Choose the detection to replace.
6. Download the generated proof image.

When **Fit ad to billboard shape** is enabled, the app tries to find the billboard's true four-corner shape inside the YOLO bounding box. If it cannot find a clean quadrilateral, it uses the rectangle fallback.

## Video Workflow

1. Upload the ad creative.
2. Upload a scene video.
3. Adjust detection confidence if needed.
4. Click **Run Video Tracking**.
5. The app processes the video with YOLO + ByteTrack.
6. Download the generated MP4 proof.

The video endpoint chooses the largest detected billboard as the initial target, then follows that same ByteTrack `track_id` across frames. If the track disappears briefly, the app tries to reconnect to the closest detection near the previous box.

## API Endpoints

### Health

```http
GET /api/health
```

Returns app and model status.

### Detect Billboards In Image

```http
POST /api/detect
```

Form fields:

- `scene`: image file
- `conf`: detection confidence, default `0.25`

Returns:

- detected boxes
- detected/fallback quadrilateral points
- preview image as a data URL
