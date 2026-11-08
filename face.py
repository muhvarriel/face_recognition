from __future__ import annotations

import argparse
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import face_recognition

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CACHE_PATH = Path(".face_cache.pkl")


def clean_person_name(file_path: Path) -> str:
    """Extract a clean person label from an image filename (e.g. 'chaeyoung~1.jpg' -> 'chaeyoung')."""
    return re.sub(r"~\d+$", "", file_path.stem)


def bgr_to_rgb_contiguous(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR array into a C-contiguous uint8 RGB array for dlib 20+ / NumPy 2+."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def load_known_faces(
    faces_dir: Path,
    cache_path: Path = DEFAULT_CACHE_PATH,
    use_cache: bool = True,
    model: str = "hog",
) -> tuple[list[np.ndarray], list[str]]:
    """Load and encode reference faces from `faces_dir`, using a disk cache when valid."""
    if not faces_dir.exists():
        raise FileNotFoundError(f"Faces directory not found: {faces_dir}")

    image_files = sorted(
        p for p in faces_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No reference images found in {faces_dir}")

    fingerprint = {str(p.resolve()): p.stat().st_mtime_ns for p in image_files}

    if use_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if (
                cached.get("fingerprint") == fingerprint
                and cached.get("model") == model
                and isinstance(cached.get("encodings"), list)
                and isinstance(cached.get("names"), list)
            ):
                return cached["encodings"], cached["names"]
        except Exception as exc:
            print(f"[warn] Ignoring unreadable cache ({exc}); rebuilding encodings...", file=sys.stderr)

    faces_encodings: list[np.ndarray] = []
    faces_names: list[str] = []

    for img_path in image_files:
        image = face_recognition.load_image_file(str(img_path))
        image = np.ascontiguousarray(image, dtype=np.uint8)
        encodings = face_recognition.face_encodings(image, model="small")
        if not encodings:
            print(f"[warn] No face detected in '{img_path.name}', skipping.", file=sys.stderr)
            continue

        faces_encodings.append(encodings[0])
        faces_names.append(clean_person_name(img_path))

    if use_cache and faces_encodings:
        try:
            with cache_path.open("wb") as f:
                pickle.dump(
                    {
                        "fingerprint": fingerprint,
                        "model": model,
                        "encodings": faces_encodings,
                        "names": faces_names,
                    },
                    f,
                )
        except OSError as exc:
            print(f"[warn] Could not write encoding cache: {exc}", file=sys.stderr)

    return faces_encodings, faces_names


def recognize_faces_in_frame(
    frame: np.ndarray,
    known_encodings: Sequence[np.ndarray],
    known_names: Sequence[str],
    scale: float = 0.25,
    tolerance: float = 0.6,
    model: str = "hog",
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Detect and match faces in a BGR frame, returning (locations_in_small_frame, matched_names)."""
    if scale != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    else:
        small_frame = frame

    rgb_small_frame = bgr_to_rgb_contiguous(small_frame)
    face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names: list[str] = []
    known_encodings_list = list(known_encodings)

    for face_encoding in face_encodings:
        name = "Unknown"
        if known_encodings_list:
            matches = face_recognition.compare_faces(
                known_encodings_list, face_encoding, tolerance=tolerance
            )
            face_distances = face_recognition.face_distance(known_encodings_list, face_encoding)
            best_match_index = int(np.argmin(face_distances))
            if matches[best_match_index]:
                name = known_names[best_match_index]
        face_names.append(name)

    return face_locations, face_names


def annotate_frame(
    frame: np.ndarray,
    face_locations: Sequence[tuple[int, int, int, int]],
    face_names: Sequence[str],
    scale: float = 0.25,
) -> np.ndarray:
    """Draw bounding boxes and name labels onto the BGR frame."""
    inv_scale = 1.0 / scale if scale > 0 else 1.0
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top = int(round(top * inv_scale))
        right = int(round(right * inv_scale))
        bottom = int(round(bottom * inv_scale))
        left = int(round(left * inv_scale))

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame


def run_image_mode(
    image_path: Path,
    known_encodings: Sequence[np.ndarray],
    known_names: Sequence[str],
    output_path: Path | None = None,
    tolerance: float = 0.6,
    model: str = "hog",
) -> list[str]:
    """Run face recognition on a single image file."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image file: {image_path}")

    face_locations, face_names = recognize_faces_in_frame(
        frame,
        known_encodings,
        known_names,
        scale=1.0,
        tolerance=tolerance,
        model=model,
    )
    annotate_frame(frame, face_locations, face_names, scale=1.0)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)

    return face_names


def run_webcam_mode(
    camera_index: int,
    known_encodings: Sequence[np.ndarray],
    known_names: Sequence[str],
    scale: float = 0.25,
    tolerance: float = 0.6,
    model: str = "hog",
) -> None:
    """Run live webcam face recognition loop (press 'q' to exit)."""
    vid = cv2.VideoCapture(camera_index)
    if not vid.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    face_locations: list[tuple[int, int, int, int]] = []
    face_names: list[str] = []
    process_this_frame = True

    try:
        while True:
            ret, frame = vid.read()
            if not ret or frame is None:
                print("[warn] Failed to grab frame from camera; exiting.", file=sys.stderr)
                break

            if process_this_frame:
                face_locations, face_names = recognize_faces_in_frame(
                    frame,
                    known_encodings,
                    known_names,
                    scale=scale,
                    tolerance=tolerance,
                    model=model,
                )

            process_this_frame = not process_this_frame

            annotate_frame(frame, face_locations, face_names, scale=scale)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        vid.release()
        cv2.destroyAllWindows()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time & static face recognition using OpenCV and dlib.")
    parser.add_argument(
        "--faces-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "faces",
        help="Directory containing known face reference images (default: data/faces)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for live mode (default: 0)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run recognition on a single image instead of live webcam",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path to save annotated image when using --image",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Face distance matching tolerance, lower is stricter (default: 0.6)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Frame downscale factor during live recognition (default: 0.25)",
    )
    parser.add_argument(
        "--model",
        choices=("hog", "cnn"),
        default="hog",
        help="Face detection model: 'hog' (CPU fast) or 'cnn' (accurate) (default: hog)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving cached reference encodings",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    known_encodings, known_names = load_known_faces(
        faces_dir=args.faces_dir,
        use_cache=not args.no_cache,
        model=args.model,
    )
    print(f"Loaded {len(known_encodings)} known face(s) from {args.faces_dir}")

    if args.image is not None:
        matched = run_image_mode(
            image_path=args.image,
            known_encodings=known_encodings,
            known_names=known_names,
            output_path=args.output,
            tolerance=args.tolerance,
            model=args.model,
        )
        print(f"Detected {len(matched)} face(s): {', '.join(matched) if matched else 'none'}")
        return 0

    run_webcam_mode(
        camera_index=args.camera,
        known_encodings=known_encodings,
        known_names=known_names,
        scale=args.scale,
        tolerance=args.tolerance,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
