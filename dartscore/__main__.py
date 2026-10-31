"""Command-line entry point:  python -m dartscore <command> --help"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dartscore", description="Automatic dartboard scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("web", help="run the web dashboard (manual clicks and/or a camera)")
    w.add_argument("--source", help="camera source: webcam, webcam:1, synthetic, synthetic:seed=3, video.mp4")
    w.add_argument("--calib", help="calibration JSON for the camera")
    w.add_argument("--auto-calibrate", action="store_true", help="auto-calibrate from the first frame")
    w.add_argument("--game", default="501")
    w.add_argument("--players", default="Player 1,Player 2")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=5000)

    c = sub.add_parser("calibrate", help="calibrate a camera/image (click 4 points, or --auto)")
    c.add_argument("--source", required=True, help="image, video, webcam, webcam:N, synthetic")
    c.add_argument("--out", default="calibration.json")
    c.add_argument("--auto", action="store_true", help="detect the board automatically")

    k = sub.add_parser("click", help="click on an image to score points (tests calibration)")
    k.add_argument("--source", required=True)
    k.add_argument("--calib", help="calibration JSON (default: auto-calibrate)")

    d = sub.add_parser("detect", help="detect a dart from before/after photos")
    d.add_argument("before")
    d.add_argument("after")
    d.add_argument("--calib", help="calibration JSON (default: auto-calibrate on BEFORE)")
    d.add_argument("--save", help="write an annotated image here")

    s = sub.add_parser("synth", help="generate sample board images with ground truth")
    s.add_argument("--out", default="samples")
    s.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)
    return {"web": cmd_web, "calibrate": cmd_calibrate, "click": cmd_click,
            "detect": cmd_detect, "synth": cmd_synth}[args.cmd](args)


def _calibration(frame, path: str | None):
    from .board_finder import find_board
    from .calibration import Calibration
    if path:
        return Calibration.load(path)
    res = find_board(frame)
    print(f"auto-calibrated: {res.correspondences} features, median residual {res.residual_mm:.2f} mm")
    return res.calibration


def cmd_web(args):
    from .game import make_game
    from .impacts import CameraImpactSource
    from .session import Session
    from .ui.web import create_app

    players = [p.strip() for p in args.players.split(",") if p.strip()]
    session = Session(make_game(args.game, players))
    camera = None
    if args.source:
        from .capture import open_source
        frames = open_source(args.source)
        first = frames.read()
        if first is None:
            sys.exit(f"no frames from {args.source}")
        if not args.calib and not (args.auto_calibrate or args.source.startswith("synthetic")):
            sys.exit("a camera source needs --calib FILE or --auto-calibrate "
                     "(run `python -m dartscore calibrate --source ...` first)")
        cal = _calibration(first, args.calib)
        camera = CameraImpactSource(frames, cal)
        session.attach(camera, interval=0.08 if args.source.startswith("synthetic") else 0.01)
    app = create_app(session, camera)
    print(f"open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


def _first_frame(spec: str):
    from .capture import open_source
    src = open_source(spec)
    for _ in range(10):   # webcams often return a few dark frames first
        f = src.read()
    src.close()
    if f is None:
        sys.exit(f"no frame from {spec}")
    return f


def cmd_calibrate(args):
    from .calibration import Calibration
    from .geometry import CALIBRATION_POINTS
    from .render import draw_board_overlay

    frame = _first_frame(args.source)
    if args.auto:
        cal = _calibration(frame, None)
    else:
        names = list(CALIBRATION_POINTS)
        clicks: dict[str, tuple[float, float]] = {}
        win = "calibrate"
        print("Click where the OUTER double wire meets these segment wires, in order:")
        for n in names:
            print(f"  - between {n.replace('/', ' and ')}")
        print("(Backspace = undo, Esc = quit)")

        def on_mouse(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(names):
                clicks[names[len(clicks)]] = (x, y)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
        while len(clicks) < len(names):
            vis = frame.copy()
            for name, (x, y) in clicks.items():
                cv2.circle(vis, (x, y), 6, (0, 255, 255), 2)
                cv2.putText(vis, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            nxt = names[len(clicks)]
            cv2.putText(vis, f"Click: double wire between {nxt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(vis, f"Click: double wire between {nxt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow(win, vis)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return 1
            if key == 8 and clicks:
                clicks.pop(list(clicks)[-1])
        cal = Calibration.from_reference_clicks(clicks)
        cv2.destroyWindow(win)
    cal.image_size = (frame.shape[1], frame.shape[0])
    vis = draw_board_overlay(frame, cal)
    cv2.imshow("calibration result: s = save, any other key = discard", vis)
    if (cv2.waitKey(0) & 0xFF) == ord("s"):
        cal.save(args.out)
        print(f"saved {args.out}")
    cv2.destroyAllWindows()


def cmd_click(args):
    from .render import draw_board_overlay, draw_hit
    from .scoring import score_xy

    frame = _first_frame(args.source)
    cal = _calibration(frame, args.calib)
    base = draw_board_overlay(frame, cal)
    vis = base.copy()

    def on_mouse(ev, x, y, *_):
        nonlocal vis
        if ev == cv2.EVENT_LBUTTONDOWN:
            bx, by = cal.image_to_board([(x, y)])[0]
            t = score_xy(bx, by)
            print(f"pixel ({x},{y}) -> board ({bx:.1f}, {by:.1f}) mm -> {t.label} = {t.points}")
            draw_hit(vis, (x, y), t.label)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            vis = base.copy()

    win = "click to score (right-click clears, Esc quits)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while (cv2.waitKey(30) & 0xFF) != 27:
        cv2.imshow(win, vis)
    cv2.destroyAllWindows()


def cmd_detect(args):
    from .detection import DiffDetector
    from .render import draw_board_overlay, draw_hit
    from .scoring import score_xy

    before, after = cv2.imread(args.before), cv2.imread(args.after)
    if before is None or after is None:
        sys.exit("could not read images")
    cal = _calibration(before, args.calib)
    det = DiffDetector().detect(before, after)
    if det is None:
        print("no dart found")
        return 1
    bx, by = cal.image_to_board([det.tip])[0]
    t = score_xy(bx, by)
    print(json.dumps({"tip_px": det.tip, "board_mm": [round(bx, 1), round(by, 1)],
                      "confidence": round(det.confidence, 2), **t.to_dict()}, default=float))
    if args.save:
        vis = draw_board_overlay(after, cal)
        draw_hit(vis, det.tip, t.label)
        cv2.imwrite(args.save, vis)


def cmd_synth(args):
    from .capture.synthetic import SyntheticSource

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    views = {"front": (0, 0, 0), "left_high": (-30, 25, 0), "right_low": (30, -20, 5), "steep": (10, 45, -8)}
    truth = {}
    for name, (az, el, roll) in views.items():
        src = SyntheticSource(seed=args.seed, azimuth=az, elevation=el, roll=roll, auto=False)
        cv2.imwrite(str(out / f"{name}_empty.png"), src.read())
        darts = []
        for i in range(3):
            r, a = rng.uniform(0, 172), rng.uniform(0, 360)
            x, y = r * np.sin(np.radians(a)), r * np.cos(np.radians(a))
            t = src.throw_at(x, y)
            cv2.imwrite(str(out / f"{name}_dart{i + 1}.png"), src.read())
            darts.append({"x": round(x, 2), "y": round(y, 2), "label": t.label, "points": t.points})
        truth[name] = {"camera": {"azimuth": az, "elevation": el, "roll": roll},
                       "H_image_to_board": src.cam.ground_truth_calibration().H.tolist(), "darts": darts}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {len(views) * 4} images + ground_truth.json to {out}/")


if __name__ == "__main__":
    sys.exit(main())
