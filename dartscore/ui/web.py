"""Flask web dashboard: scoreboard, clickable board, optional live camera view."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

from ..geometry import SEGMENT_ORDER, STANDARD
from ..impacts import CameraImpactSource, Impact
from ..render import draw_board_overlay, draw_hit
from ..scoring import parse_throw, score_xy
from ..session import Session

STATIC = Path(__file__).parent / "static"


def create_app(session: Session, camera: CameraImpactSource | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/geometry")
    def geometry():
        return jsonify({"dims": STANDARD.__dict__, "order": SEGMENT_ORDER, "camera": camera is not None})

    @app.get("/api/state")
    def state():
        return jsonify(session.state())

    @app.post("/api/impact")
    def impact():
        d = request.get_json(force=True)
        session.handle(Impact(float(d["x"]), float(d["y"]), source="click"))
        return jsonify(session.state())

    @app.post("/api/throw")
    def throw():
        try:
            t = parse_throw(request.get_json(force=True)["label"])
        except (ValueError, KeyError, IndexError) as e:
            return jsonify({"error": str(e)}), 400
        session.throw(t, "keyboard")
        return jsonify(session.state())

    @app.get("/api/score")
    def score():
        t = score_xy(float(request.args["x"]), float(request.args["y"]))
        return jsonify(t.to_dict())

    @app.post("/api/undo")
    def undo():
        session.undo()
        return jsonify(session.state())

    @app.post("/api/next")
    def next_turn():
        session.next_turn()
        return jsonify(session.state())

    @app.post("/api/new")
    def new_game():
        d = request.get_json(force=True)
        players = [p.strip() for p in d.get("players", "Player 1").split(",") if p.strip()] or ["Player 1"]
        kind = str(d.get("game", "501"))
        opts = {} if kind == "cricket" else {"double_out": bool(d.get("double_out", True))}
        session.new_game(kind, players, **opts)
        return jsonify(session.state())

    @app.get("/video.mjpg")
    def video():
        if camera is None:
            return "no camera", 404

        def frames():
            while True:
                f = camera.last_frame
                if f is not None:
                    img = draw_board_overlay(f, camera.cal, (0, 220, 255))
                    for uv in list(camera.hits):
                        u, v = camera.cal.image_to_board([uv])[0]
                        draw_hit(img, uv, score_xy(u, v).label)
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                time.sleep(0.1)

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app
