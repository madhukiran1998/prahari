"""Clip encoding actually has to produce a playable file.

This exists because a wrong keyword argument to the encoder failed only at
runtime, ten seconds after an alert, in a background thread - the one place
nobody is watching during a demo.
"""

import cv2
import numpy as np
import pytest

from prahari.clips import write_clip


def jpegs(count: int, width: int = 640, height: int = 360) -> list[bytes]:
    out = []
    for i in range(count):
        frame = np.full((height, width, 3), i * 7 % 255, np.uint8)
        cv2.rectangle(frame, (i * 4, 40), (i * 4 + 60, 200), (0, 0, 255), -1)
        ok, buf = cv2.imencode(".jpg", frame)
        assert ok
        out.append(buf.tobytes())
    return out


def test_writes_a_playable_mp4(tmp_path):
    path = tmp_path / "event.mp4"
    result = write_clip(path, jpegs(25), fps=5)

    assert result == path
    assert path.exists() and path.stat().st_size > 0

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), "encoder produced a file OpenCV cannot open"
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (360, 640)


def test_no_frames_is_not_a_crash(tmp_path):
    assert write_clip(tmp_path / "empty.mp4", [], fps=5) is None


def test_undecodable_frames_are_skipped(tmp_path):
    path = tmp_path / "mixed.mp4"
    frames = jpegs(10)
    frames.insert(5, b"not a jpeg")
    assert write_clip(path, frames, fps=5) == path
    assert path.exists()


def test_odd_sized_frames_are_resized_to_the_first(tmp_path):
    """A camera that reconnects at a different resolution must not kill the clip."""
    path = tmp_path / "resize.mp4"
    frames = jpegs(6) + jpegs(4, width=320, height=180)
    assert write_clip(path, frames, fps=5) == path
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    cap.release()


@pytest.mark.parametrize("fps", [1, 5, 15])
def test_accepts_the_frame_rates_we_actually_use(tmp_path, fps):
    path = tmp_path / f"fps{fps}.mp4"
    assert write_clip(path, jpegs(12), fps=fps) == path
