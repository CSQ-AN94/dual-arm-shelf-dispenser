"""Headless browser dashboard and stop control."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import cv2

from .core import Detection


class SharedState:
    def __init__(self, stop_event: threading.Event):
        self.lock = threading.Lock()
        self.stop_event = stop_event
        self.stage = "初始化"
        self.message = ""
        self.jpeg: Optional[bytes] = None
        self.detection: Optional[Detection] = None
        self.depth_m: Optional[float] = None

    def update(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def status(self) -> dict:
        with self.lock:
            return {
                "stage": self.stage,
                "message": self.message,
                "depth_m": self.depth_m,
                "stopped": self.stop_event.is_set(),
            }


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            body = b"""<!doctype html><meta charset=utf-8><title>Bottle grasp demo</title>
<style>body{margin:0;background:#111;color:#eee;font:16px sans-serif;text-align:center}
img{max-width:100vw;max-height:78vh}button{font-size:20px;padding:10px 28px;background:#b22;color:white;border:0}
#s{padding:8px}</style><h2>Right wrist bottle grasp</h2><div id=s></div>
<img src=/stream.mjpg><p><button onclick=fetch('/stop',{method:'POST'})>STOP AND HOLD</button></p>
<script>setInterval(async()=>{let x=await(await fetch('/status.json')).json();s.textContent=x.stage+' | '+x.message+(x.depth_m?' | '+x.depth_m.toFixed(3)+' m':'')},500)</script>"""
            self._send("text/html; charset=utf-8", body)
        elif self.path == "/status.json":
            self._send(
                "application/json",
                json.dumps(self.server.state.status()).encode(),
            )
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while not self.server.shutdown_event.is_set():
                    with self.server.state.lock:
                        jpeg = self.server.state.jpeg
                    if jpeg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/stop":
            self.server.state.stop_event.set()
            self._send("application/json", b'{"ok":true}')
        else:
            self.send_error(404)

    def _send(self, content_type: str, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


class Dashboard:
    def __init__(self, state: SharedState, host: str, port: int):
        self.server = ThreadingHTTPServer((host, port), DashboardHandler)
        self.server.state = state
        self.server.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown_event.set()
        self.server.shutdown()
        self.server.server_close()


class PreviewWorker(threading.Thread):
    def __init__(self, camera: Any, state: SharedState):
        super().__init__(daemon=True)
        self.camera = camera
        self.state = state
        self.local_stop = threading.Event()

    def run(self):
        while not self.state.stop_event.is_set() and not self.local_stop.is_set():
            color, _ = self.camera.get_latest_frames()
            if color is None:
                time.sleep(0.05)
                continue
            with self.state.lock:
                det = self.state.detection
                stage = self.state.stage
                message = self.state.message
            frame = color.copy()
            if det:
                x1, y1, x2, y2 = det.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
                cv2.putText(
                    frame,
                    f"{det.class_name} {det.confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (40, 220, 40),
                    2,
                )
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (0, 0, 0), -1)
            cv2.putText(
                frame, stage, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2
            )
            cv2.putText(
                frame,
                message[:70],
                (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            if ok:
                self.state.update(jpeg=encoded.tobytes())
            time.sleep(0.05)

    def stop(self):
        self.local_stop.set()
