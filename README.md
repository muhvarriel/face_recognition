# Face Recognition Engine

## Abstract

Face Recognition Engine is a deterministic computer vision application for real-time video stream and static image biometric identification. It resolves the latency and memory-alignment bottlenecks common to Python-based facial recognition pipelines by combining spatial frame decimation, temporal frame striding, C-contiguous RGB buffer normalization for `dlib` 20.x and `numpy` 2.x interoperability, and a self-invalidating disk cache for 128-dimensional facial embeddings.

---

## Architecture

### System Design

The system implements a modular procedural pipeline supporting two execution sinks: interactive video stream processing (`run_webcam_mode`) and headless batch image evaluation (`run_image_mode`).

```text
+---------------------------------------------------------------------------+
|                        1. INITIALIZATION PHASE                            |
|                                                                           |
|  [data/faces/*] ---> [Nanosecond mtime Fingerprint]                       |
|                             |                                             |
|                             v                                             |
|                    (.face_cache.pkl valid?)                               |
|                       /                \                                  |
|                 Yes  /                  \  No                             |
|                     v                    v                                |
|          [Load Cached Vectors]   [dlib ResNet-34 Encoder]                 |
|                     \                    /                                |
|                      v                  v                                 |
|                [In-Memory 128-D Gallery Matrix]                           |
+---------------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------------+
|                         2. INFERENCE PIPELINE                             |
|                                                                           |
|  [BGR Frame Input] ---> [Spatial Downscale (--scale)]                     |
|                                 |                                         |
|                                 v                                         |
|                 [BGR-to-RGB C-Contiguous Buffer Adapter]                  |
|                                 |                                         |
|                                 v                                         |
|                 [Face Localization (HOG / CNN Detector)]                  |
|                                 |                                         |
|                                 v                                         |
|                 [68-Point Landmark & 128-D Extraction]                    |
|                                 |                                         |
|                                 v                                         |
|                 [Euclidean (L2) Distance Minimization]                    |
|                                 |                                         |
|                                 v                                         |
|                 [Coordinate Projection & BGR Annotation]                  |
+---------------------------------------------------------------------------+
```

### Core Engineering Patterns

1. **Content-Fingerprinted Read-Through Cache**: At startup, `load_known_faces()` scans the reference gallery directory and constructs a metadata fingerprint mapping each resolved file path to its nanosecond modification timestamp (`st_mtime_ns`). If `.face_cache.pkl` matches both the directory fingerprint and the selected detection model, precomputed 128-D descriptors are deserialized directly into memory, reducing cold-start initialization from $O(N)$ deep-network forward passes to a single disk read.
2. **C-Contiguous Memory Boundary Adaptation**: OpenCV represents frames in BGR channel order, and negative-stride array slices (`frame[:, :, ::-1]`) produce non-contiguous memory views that violate the pybind11 buffer protocol enforced by `dlib` 20.x and `numpy` 2.x. The `bgr_to_rgb_contiguous()` adapter executes `cv2.cvtColor()` followed by `np.ascontiguousarray(..., dtype=np.uint8)` to guarantee C-contiguous alignment prior to native C++ invocation.
3. **Spatial-Temporal Decimation**: During live video capture, frames are downscaled by a configurable factor (`--scale 0.25` by default, yielding a $16\times$ reduction in pixel throughput) and evaluated on alternating frames (`process_this_frame` toggle). Bounding boxes are scaled inversely (`1.0 / scale`) prior to rendering onto the native-resolution output buffer.
4. **Deterministic Resource Management**: Hardware camera descriptors (`cv2.VideoCapture`) and HighGUI window contexts are enclosed in a `try...finally` block to guarantee release upon standard termination or unhandled interrupts.

### Technology Stack

| Layer | Package | Version Constraint | Verified Version | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| Runtime | Python (CPython) | `>=3.10, <3.15` | `3.12.14` | Execution environment |
| Native Vision Engine | `dlib` | `>=20.0.1` | `20.0.1` | C++ HOG/MMOD face detection and ResNet-34 metric learning |
| Recognition API | `face-recognition` | `>=1.3.0` | `1.3.0` | High-level Python bindings over `dlib` primitives |
| Model Weights | `face-recognition-models` | `>=0.3.0` | `0.3.0` | Pretrained landmark predictor and ResNet-34 weight blobs |
| Computer Vision & I/O | `opencv-python` | `>=5.0.0` | `5.0.0.93` | Video capture abstraction, image I/O, color transformation, and overlay rendering |
| Numerical Computing | `numpy` | `>=2.0.0` | `2.5.4` | Contiguous tensor layout enforcement and Euclidean distance vectorization |
| Image Decoding | `Pillow` | `>=11.0.0` | `13.0.0` | Reference image decoding |
| Packaging Runtime | `setuptools` | `<81` | `80.10.2` | Retains `pkg_resources` API required by `face-recognition-models` |

---

## Prerequisites

### Operating System & Toolchain

- **Operating System**: macOS 13+ (ARM64 / x86_64), Linux (glibc 2.31+), or Windows 10/11 with Visual Studio C++ Build Tools.
- **Python**: CPython `3.10` through `3.14` (`3.12` recommended).
- **Build Dependencies**:
  - `cmake >= 3.18` (required to compile `dlib` from source when platform wheels are unavailable).
  - C++17 compliant compiler (`clang++` via Xcode Command Line Tools on macOS, `g++ >= 11` on Linux, or MSVC v143 on Windows).
- **Hardware**:
  - Minimum: Dual-core CPU, 4 GB RAM.
  - Live Mode: USB or integrated UVC-compliant camera device.
  - Optional: CUDA-capable GPU (when `dlib` is compiled with `-DDLIB_USE_CUDA=1` for `--model cnn`).

Verify build toolchain availability prior to installation:

```bash
cmake --version
c++ --version
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/muhvarriel/face_recognition.git
cd face_recognition
```

### 2. Provision the Virtual Environment

#### Option A: Using `uv` (Recommended)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

#### Option B: Using Standard `venv` and `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

The application is configured strictly via command-line flags and filesystem conventions. It does not require external `.env` files.

### CLI Configuration Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--faces-dir` | `Path` | `data/faces` | Directory containing reference face images (`.jpg`, `.jpeg`, `.png`, `.webp`). |
| `--camera` | `int` | `0` | Zero-based hardware camera device index for live stream capture. |
| `--image` | `Path` | `None` | Path to a static input image. Bypasses webcam capture and executes single-image evaluation. |
| `--output` | `Path` | `None` | Output file path to persist the annotated image when `--image` is specified. |
| `--tolerance` | `float` | `0.6` | Maximum Euclidean ($L_2$) distance threshold for positive identity classification. Lower values enforce stricter matching. |
| `--scale` | `float` | `0.25` | Spatial downscaling factor `(0.0, 1.0]` applied prior to face localization in live webcam mode. |
| `--model` | `str` | `hog` | Face localization backend: `hog` (CPU-optimized Histogram of Oriented Gradients) or `cnn` (MMOD Convolutional Neural Network). |
| `--no-cache` | `bool` | `False` | Disables reading and writing `.face_cache.pkl`, forcing full descriptor recomputation. |

### Reference Gallery Naming Convention

Place reference portraits in `data/faces/`. The label parser (`clean_person_name`) derives the subject identifier from the filename stem and strips trailing numeric tilde suffixes (for example, `data/faces/alex.jpg` resolves to `alex`, and `data/faces/name~1.jpg` resolves to `name`). Each reference image should contain a single unobstructed frontal face.

---

## Usage

### 1. Live Webcam Recognition

Launch the interactive recognition loop using the default camera device:

```bash
.venv/bin/python face.py
```

Press `q` while the display window is focused to terminate capture and release hardware handles.

### 2. Static Image Evaluation and Annotation

Execute recognition against a single static image and write the annotated output frame to disk:

```bash
.venv/bin/python face.py --image data/faces/alex.jpg --output build/alex_annotated.jpg
```

Expected standard output:

```text
Loaded 10 known face(s) from /path/to/face_recognition/data/faces
Detected 1 face(s): alex
```

### 3. High-Precision Matching with Custom Tolerance and Cache Bypass

Run static evaluation with a stricter $L_2$ distance threshold (`0.50`) and force descriptor recomputation:

```bash
.venv/bin/python face.py \
  --faces-dir data/faces \
  --image data/faces/sarah.jpg \
  --tolerance 0.50 \
  --no-cache
```

---

## Testing

### 1. CLI Contract Verification

Verify argument parser integrity and exit status:

```bash
.venv/bin/python face.py --help
```

### 2. Reference Gallery Encoding & Cache Invalidation Test

Execute a cold-cache pass followed by a warm-cache pass to verify that all 10 reference images encode cleanly and serialize to `.face_cache.pkl`:

```bash
.venv/bin/python face.py --no-cache --image data/faces/alex.jpg
.venv/bin/python face.py --image data/faces/sarah.jpg
```

### 3. Batch End-to-End Identity Verification

Run an automated verification sweep across all images in `data/faces/` to assert that every reference subject is detected and accurately classified against the gallery:

```bash
.venv/bin/python -c "
from pathlib import Path
import face

faces_dir = Path('data/faces')
encodings, names = face.load_known_faces(faces_dir, use_cache=True)
assert len(encodings) == 10, f'Expected 10 encodings, got {len(encodings)}'

for img_path in sorted(faces_dir.glob('*.jpg')):
    expected = face.clean_person_name(img_path)
    matched = face.run_image_mode(img_path, encodings, names)
    assert matched == [expected], f'Mismatch on {img_path.name}: expected [{expected}], got {matched}'
print('All 10 reference identities verified successfully.')
"
```

---

## Project Structure

```text
face_recognition/
|-- .gitignore               # Git exclusion rules for virtualenvs, bytecode, and cache artifacts
|-- README.md                # System architecture, operational guide, and API documentation
|-- requirements.txt         # Pinned dependency manifest for reproducible builds
|-- face.py                  # Core CLI entrypoint, encoding cache manager, and inference pipeline
`-- data/
    `-- faces/               # Reference facial gallery (10 public domain / Unsplash portraits)
        |-- alex.jpg
        |-- daniel.jpg
        |-- david.jpg
        |-- emily.jpg
        |-- grace.jpg
        |-- james.jpg
        |-- michael.jpg
        |-- olivia.jpg
        |-- sarah.jpg
        `-- sophia.jpg
```

---

## Contributing

1. **Branching Strategy**: Create feature or defect branches from `main` using the naming convention `<type>/<short-description>` (for example, `feat/batch-directory-inference` or `fix/camera-index-validation`).
2. **Code Standards**:
   - Target Python `3.10+` syntax with `from __future__ import annotations` enabled.
   - Provide explicit PEP 484 type annotations for all function signatures and return values.
   - Ensure all OpenCV image buffers passed to `dlib` or `face_recognition` are converted via `bgr_to_rgb_contiguous()`.
   - Do not commit `.face_cache.pkl`, virtual environments, or personally identifiable biometric datasets without explicit consent and license clearance.
3. **Verification Requirement**: Execute the batch identity verification script in the **Testing** section and confirm zero regressions prior to opening a pull request.

---

## License

This repository currently contains no explicit `LICENSE` file; under standard copyright law, all rights are reserved by the repository owner unless otherwise specified. Reference portrait images in `data/faces/` are sourced from Unsplash and are governed by the [Unsplash License](https://unsplash.com/license).
