# keepy-ups-counter

Detects a football and the player's pose in a keepy-ups video, then renders an
annotated output with a live signal panel.

## How it works

The pipeline runs per frame ([src/pipeline.py](src/pipeline.py)):

1. **Ball detection** — a finetuned YOLO ONNX model (`yolo11` or `yolo26`) detects the
   ball via `onnxruntime`. 
2. **Pose estimation** — MediaPipe Pose extracts the player's skeleton,
   including the foot-index landmarks used as toe positions.
3. **Ball tracking** — a centroid + physics-informed Kalman tracker associates
   detections across frames. It runs in metric space: pixels-per-metre is
   auto-estimated from the ball's pixel diameter and a known real-world ball
   diameter (default 0.22 m), so gravity stays a constant 9.81 m/s² regardless
   of camera distance. The tracker keeps a state estimate alive through brief
   occlusions.
4. **Counting** — a 3-state machine on the ball's vertical velocity
   ([src/keepy_ups_counter.py](src/keepy_ups_counter.py)) counts a keepy-up on
   each `FALLING → RISING` reversal, but only when a valid body part (head,
   shoulders, knees, feet) is within ~ball-radius of the ball. Arms are excluded
   so hand touches don't count; ground bounces are rejected by the same
   proximity check. A short frame cooldown prevents vy noise from
   double-counting one flick.
5. **Rendering** — boxes, skeleton, Kalman predictions and a diagnostic HUD are
   drawn on the frame. When writing/displaying, a scrolling 8-channel signal
   panel (ball x/y/vx/vy + left/right toe x/y) is concatenated to the right.

## Run

```bash
python main.py --video path/to/clip.mp4 --output output/ --display
```

Key flags (see `python main.py --help` for the full list):

- `--ball-model` — path to the YOLO ONNX (default `models/yolo11n.onnx`)
- `--ball-arch` — `yolo11` (transposed + NMS) or `yolo26` (NMS-free)
- `--no-ball-tracking` — skip the Kalman tracker, use raw per-frame detections
- `--ball-diameter-m` — real-world ball size used for px/m auto-estimation
- `--display` — show frames in a window (press `q` to quit)
- `--output <dir>` — write `annotated_<timestamp>.mp4` into the given directory
