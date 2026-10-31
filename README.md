# dartscore: automatic dartboard scoring

Camera-based dart scoring, built to be developed **without hardware first**.
A simulated camera renders a regulation board from any angle with darts being
thrown and pulled out. That exercises the full pipeline (calibration, dart
detection, scoring, game) against exact ground truth. A webcam, video file or
photo plugs into the same pipeline, and so does a future multi-camera rig or
sensor board.

## Quick start (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest            # 73 tests, ~20 s
```

**See the whole thing running (no hardware):**

```powershell
.venv\Scripts\python -m dartscore web --source synthetic
```

Open http://127.0.0.1:5000. A simulated player throws darts at a virtual board
filmed from an angle. Each dart is auto-detected and scored into a 501 game.
When a "hand" pulls the darts out, the turn passes to the next player. The
camera panel shows the live feed with the calibrated board overlay and the
detected hits.

**Manual scoring dashboard (click the board where the dart landed):**

```powershell
.venv\Scripts\python -m dartscore web
```

You can also type throws (`T20`, `D16`, `25`, `bull`, `miss`), undo, and start
501/301/701/Cricket games for any number of players.

## Architecture

```
 FrameSource ──► DartDetector ──► Calibration ──┐
 (webcam, file,   (tip pixel)     (pixel → mm)  │
  simulator)                                    ▼
                                    ┌──── ImpactSource ────┐   ◄── the hardware seam:
 other cameras ──► … ──► FusedImpactSource    │   events are in BOARD mm,
 contact sensors / e-board ──► (your source)  │   never pixels
                                    └──────────┬───────────┘
                                               ▼
                                   Scoring engine (pure maths)
                                               ▼
                                   Game (X01, Cricket) ◄── Session ◄── Web UI
```

| Module | Role | Swap it by… |
|---|---|---|
| [`capture/`](dartscore/capture) | `FrameSource.read()` → BGR frame. Webcam, image, video, `SyntheticSource`. | Subclass `FrameSource` (IP cam, Pi camera, IR camera) |
| [`detection/`](dartscore/detection) | `DartDetector.detect(before, after)` → tip pixel. `DiffDetector` = frame difference + "tip is the narrow end of the blob". | Subclass `DartDetector` (ML keypoint model, IR blob…) |
| [`calibration.py`](dartscore/calibration.py) | Homography pixel ↔ board mm, from 4 clicks (or more points). Save/load JSON. | One file per camera |
| [`board_finder.py`](dartscore/board_finder.py) | Auto-calibration from a single image (see below). | — |
| [`impacts.py`](dartscore/impacts.py) | `ImpactSource.poll()` → `Impact(x, y)` / `Takeout`. `CameraImpactSource` (settle/takeout state machine), `ManualImpactSource`, `FusedImpactSource`. | Subclass `ImpactSource` for non-camera hardware |
| [`scoring.py`](dartscore/scoring.py) | `score_xy(x, y)` → `Throw` (S/D/T 1–20, 25, BULL, MISS). No dependencies. | — |
| [`game/`](dartscore/game) | `X01` (double-out/in, busts, checkout hints, averages), `Cricket`. Undo everywhere. | Subclass `Game` |
| [`session.py`](dartscore/session.py), [`ui/`](dartscore/ui) | Thread-safe glue, Flask dashboard + MJPEG camera view. | Add a scoreboard display as another consumer of `session.state()` |
| [`render.py`](dartscore/render.py) | `VirtualCamera` (pinhole, any angle), board/dart renderer, overlays. | — |

**Board coordinates** (used everywhere past the calibration step): millimetres,
origin at the bull, +y towards 20, angles clockwise from 12 o'clock.
Dimensions are the regulation ones in [`geometry.py`](dartscore/geometry.py).

## Command-line tools

```powershell
.venv\Scripts\python -m dartscore synth --out samples           # sample images + ground_truth.json
.venv\Scripts\python -m dartscore calibrate --source samples\left_high_empty.png --auto
.venv\Scripts\python -m dartscore calibrate --source webcam --out calibration.json   # click 4 points
.venv\Scripts\python -m dartscore click --source photo.jpg --calib calibration.json  # click → score
.venv\Scripts\python -m dartscore detect before.jpg after.jpg --save annotated.png
.venv\Scripts\python -m dartscore web --source webcam --calib calibration.json
```

`samples/` already contains 4 camera angles × (empty + 1, 2, 3 darts), with the
true homography and dart positions in `ground_truth.json`.

## Connecting a real camera

1. **Mount it.** Fix the camera rigidly (any bump invalidates the calibration),
   ideally **off to the side or above, 20–45° off-axis**, 0.6–1 m from the board.
   Avoid looking straight at the board: from the front, a dart's flight hides its
   own tip (see accuracy below). Use even lighting with no hard shadows.
2. **Check the feed:** `python -m dartscore click --source webcam` opens your
   webcam (`webcam:1` for a second camera).
3. **Calibrate:** `python -m dartscore calibrate --source webcam --out calibration.json`.
   Click where the **outer double wire** meets the segment wires at 20|1, 6|10, 3|19, 11|14.
   The yellow overlay should then sit on the wires. Press `s` to save.
   You can try `--auto` first. It works on clearly coloured boards, assumes 20 is
   roughly at the top, and may need HSV threshold tuning in `board_finder.color_labels`
   for your lighting.
4. **Verify:** `python -m dartscore click --source webcam --calib calibration.json`
   and click around the board. Every bed should score correctly.
5. **Play:** `python -m dartscore web --source webcam --calib calibration.json`.
   Leave the board empty for the first second (it learns the empty board), throw,
   and pull the darts out after each visit. The pull-out is detected as a takeout
   and ends the turn. The **Undo** and **Next player** buttons cover misdetections.

Tuning knobs: `CameraImpactSource(settle_frames, motion_frac, change_frac)` and
`DiffDetector(threshold, group_px)`.

## Adding other hardware later

* **Multiple cameras:** one `CameraImpactSource` per camera, each with its own
  calibration, wrapped in `FusedImpactSource([...])`. Each camera already reports
  board millimetres, so fusion is just merging points. The default is a
  confidence-weighted average. A better strategy (trust the camera with the
  clearest view of that region) only replaces `FusedImpactSource._fuse`.
* **IR / depth camera:** if it delivers images, it's a new `FrameSource` and
  maybe a new `DartDetector`. If it delivers 3-D points, intersect the dart line
  with the board plane and emit `Impact`s directly.
* **Contact / piezo / electronic board:** write an `ImpactSource` that maps each
  sensor to the centre of its bed (`geometry.polar_to_xy`) and emits `Impact`.
  Scoring, games and UI don't change.
* **Physical scoreboard:** poll `session.state()` (or `GET /api/state`) and push
  it to an LED matrix, e-ink display, etc.

## Measured accuracy (synthetic data)

* **Auto-calibration:** ≤ 0.7 mm max error over the board at every angle tested,
  up to 45° off-axis with roll. The bullseye anchors the centre, and robust least
  squares fits the ring edges radially and the segment boundaries angularly.
* **4-click calibration:** exact with perfect clicks, about 1–2 mm with ±1.5 px
  click noise.
* **Dart tip, angled camera:** median ≈ 2 mm. The occasional larger error happens
  when a new dart overlaps an earlier one.
* **Dart tip, camera head-on:** often wrong by several cm, because the flight
  covers the tip. That's a geometric limit of a frontal camera, so mount the
  camera at an angle, or use 2–3 cameras around the board as commercial systems do.

Synthetic images are cleaner than reality. Expect to tune thresholds on real
footage: lighting changes, shadows of darts, and auto-exposure all matter.

## Roadmap ideas

* Record real before/after photos and add them to `tests/` as regression cases.
* Lens-distortion correction (`cv2.calibrateCamera` with a checkerboard) for wide-angle webcams.
* Tip refinement: search along the dart axis for the board-contact point.
* Multi-camera fusion that weights each camera by its viewing angle onto that board region.
* Bounce-out / missed-dart handling, more games (Around the Clock, Shanghai).
