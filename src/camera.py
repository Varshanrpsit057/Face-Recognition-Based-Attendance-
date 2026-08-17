"""
Camera capture — threaded, reconnecting source for webcams, video
files, and RTSP/HTTP IP cameras.

A single 8MP IP camera feeding a tiled detection pipeline is the
target deployment: detection+recognition on a full classroom frame
takes well over one frame interval, so a naive `cv2.VideoCapture.read()`
loop falls behind and processes an ever-growing backlog of stale
frames. This module runs capture in a background thread and always
exposes only the newest frame, and transparently reconnects an IP
camera stream that drops (Wi-Fi hiccup, camera reboot, etc.) instead
of silently freezing the pipeline.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from config import cfg
from src.logger import get_logger

logger = get_logger(__name__)


def resolve_source(source: Union[str, int, None]) -> Union[str, int]:
    """Normalize a user-provided camera source string.

    Accepts an integer/int-like string for local webcam indices, or a
    string for RTSP/HTTP URLs and video file paths.
    """
    if source is None:
        return cfg.camera.ip_camera_url or cfg.camera.source
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return s


def is_network_stream(source: Union[str, int]) -> bool:
    return isinstance(source, str) and (
        source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://")
    )


class CameraStream:
    """Threaded camera/video/IP-camera reader exposing the latest frame.

    Usage:
        cam = CameraStream("rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101")
        cam.start()
        ok, frame = cam.read()
        ...
        cam.stop()
    """

    def __init__(self, source: Union[str, int, None] = None, config: Optional[object] = None):
        self.config = config or cfg.camera
        self.source = resolve_source(source)
        self.is_network = is_network_stream(self.source)

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_ready = False
        self._running = False
        self._last_read_time = 0.0
        self._last_error: Optional[str] = None
        self._reconnect_count = 0
        self._frame_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> "CameraStream":
        if self._running:
            return self
        self._running = True
        self._open()
        if getattr(self.config, "use_threaded_capture", True):
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ── Capture internals ────────────────────────────────────────

    def _open(self) -> bool:
        self._release()

        if self.is_network:
            transport = getattr(self.config, "rtsp_transport", "tcp")
            # Must be set before VideoCapture() is constructed — OpenCV's
            # FFMPEG backend reads this env var at open() time.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{transport}|stimeout;{int(getattr(self.config, 'connect_timeout_sec', 8.0) * 1_000_000)}"
            )
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        elif isinstance(self.source, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self._last_error = f"Could not open source: {self.source}"
            logger.warning(self._last_error)
            cap.release()
            self._cap = None
            return False

        if not self.is_network:
            # A local file/webcam accepts explicit resolution/fps requests;
            # an IP camera streams whatever it's configured to on its own
            # side (setting these on an RTSP capture is usually a no-op or
            # can even stall some backends).
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, getattr(self.config, "width", 1280))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, getattr(self.config, "height", 720))
            fps = getattr(self.config, "fps", None)
            if fps:
                cap.set(cv2.CAP_PROP_FPS, fps)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, getattr(self.config, "buffer_size", 1)))

        self._cap = cap
        self._last_error = None
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _loop(self) -> None:
        reconnect_delay = getattr(self.config, "reconnect_delay_sec", 2.0)
        max_delay = getattr(self.config, "max_reconnect_delay_sec", 30.0)
        read_timeout = getattr(self.config, "read_timeout_sec", 5.0)
        delay = reconnect_delay

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.info(f"Reconnecting camera source '{self.source}' (attempt {self._reconnect_count + 1})...")
                if self._open():
                    self._reconnect_count += 1
                    delay = reconnect_delay
                    self._last_read_time = time.time()
                else:
                    time.sleep(delay)
                    delay = min(max_delay, delay * 1.5)
                continue

            ok, frame = self._cap.read()
            now = time.time()
            if not ok or frame is None:
                if now - self._last_read_time > read_timeout:
                    logger.warning(f"Camera '{self.source}' stopped producing frames — reconnecting.")
                    self._release()
                continue

            self._last_read_time = now
            self._frame_count += 1
            with self._lock:
                self._frame = frame
                self._frame_ready = True

    # ── Reading ───────────────────────────────────────────────────

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the most recent frame. Non-blocking when threaded."""
        if self._thread is not None:
            with self._lock:
                if not self._frame_ready:
                    return False, None
                return True, self._frame.copy()

        # Unthreaded fallback: synchronous read.
        if self._cap is None or not self._cap.isOpened():
            if not self._open():
                return False, None
        ok, frame = self._cap.read()
        if ok:
            self._frame_count += 1
        return ok, frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error


def make_preview(frame: np.ndarray, max_width: Optional[int] = None) -> np.ndarray:
    """Downscale a full-resolution detection frame for responsive display.
    Detection always runs on the original frame; only the on-screen
    preview is shrunk to keep the browser/UI fast."""
    max_width = max_width or getattr(cfg.camera, "preview_max_width", 1280)
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
