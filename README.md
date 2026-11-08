# Face Recognition

Real-time webcam and static image face recognition using [`face_recognition`](https://github.com/ageitgey/face_recognition) (`dlib` 20.x), OpenCV 5.x, and NumPy 2.x.

## Features

- **Real-time Webcam Recognition**: Detects and labels faces in live video (`q` to exit).
- **Static Image Mode**: Run recognition on a single image file and optionally save an annotated result (`--image` / `--output`).
- **Automatic Encoding Cache**: Caches 128-D face encodings in `.face_cache.pkl` and automatically invalidates when files in `data/faces/` are added, modified, or removed.
- **Public Reference Dataset**: Ships with 10 sample public portrait images from Unsplash in `data/faces/`.

## Setup

Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or using `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Live Webcam Mode

```bash
python face.py
```

Press `q` in the video window to quit.

### Static Image Mode

```bash
python face.py --image data/faces/alex.jpg --output out.jpg
```

### CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--faces-dir` | `data/faces` | Directory containing known reference face images (`.jpg`, `.jpeg`, `.png`, `.webp`) |
| `--camera` | `0` | Camera device index for live mode |
| `--image` | `None` | Path to a single image to run recognition on instead of webcam |
| `--output` | `None` | Output path to save the annotated image when using `--image` |
| `--tolerance` | `0.6` | Face distance matching threshold (lower is stricter) |
| `--scale` | `0.25` | Frame downscale factor for faster real-time detection |
| `--model` | `hog` | Face detection model: `hog` (fast on CPU) or `cnn` (more accurate) |
| `--no-cache` | `False` | Force recomputing reference face encodings |
