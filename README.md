# TrinetraAI

Control your computer with a webcam. Turn your head to scroll, make a hand gesture to click. No eye tracker, no special hardware, runs on CPU.

I built this for people who can't use a mouse or keyboard comfortably (ALS, cerebral palsy, spinal injuries, tremors). Eye trackers solve this but are expensive. Most people already have a webcam.

## How it works

Head movement is plain geometry: MediaPipe FaceMesh gives landmarks, `solvePnP` gives yaw/pitch, and fixed thresholds trigger scrolling. No training needed.

Hand gestures are different. Everyone's range of motion is different, so you record your own samples and train a small classifier (SVM by default, LSTM if you want gestures that depend on motion over time). A gesture only fires if confidence clears a cutoff and the debounce timer allows it.

The webcam is read in its own thread so a slow frame never stalls capture.

```
webcam -> FaceMesh -> yaw/pitch -> thresholds ---+
      \                                          +-> debounce -> pynput/PyAutoGUI
       -> Hands -> 63-d vector -> SVM/LSTM ------+
```

## Controls

Head:

- turn left (yaw > +15) scrolls left
- turn right (yaw < -15) scrolls right
- nod down (pitch > +10) scrolls down
- tilt up (pitch < -10) scrolls up

Hands (you train these):

- `open_palm` is left click
- `fist` is right click
- `peace_sign` launches an app, set in config
- `neutral` does nothing, and is also the fallback when the model isn't confident

## Setup

Python 3.10+, a webcam, and Linux, macOS, or Windows 10/11.

```bash
git clone https://github.com/chetx27/TrinetraAI.git
cd TrinetraAI
python -m venv venv
source venv/bin/activate    # windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py --config config/settings.yaml
```

Press `q` or `ESC` to quit. It prints latency stats on exit.

Want to test without it actually clicking things? Add `--no-actions`.

## Training your gestures

60 samples per class is a good start.

```bash
python -m classifier.trainer --gesture neutral --samples 60
python -m classifier.trainer --gesture open_palm --samples 60
python -m classifier.trainer --gesture fist --samples 60
python -m classifier.trainer --gesture peace_sign --samples 60
python -m classifier.trainer --fit
```

For the LSTM:

```bash
python -m classifier.trainer --fit --lstm
python main.py --config config/settings.yaml --lstm
```

## Config

Everything lives in `config/settings.yaml`:

```yaml
vision:
  camera_index: 0
  frame_width: 640
  frame_height: 480
  fps: 30

control:
  debounce_time_ms: 350
  scroll_speed: 15
  head_pose:
    yaw_threshold: 15.0
    pitch_threshold: 10.0

model:
  backend: "svm"        # or "lstm"
  confidence_cutoff: 0.85
```

Getting accidental triggers? Raise `confidence_cutoff` or `debounce_time_ms` first.

## Performance

Tested on a 10th-gen i5 laptop at 640x480. Yours will differ depending on CPU and lighting.

- end-to-end: ~28 ms (p95 ~38 ms)
- gesture accuracy: ~95% on held-out samples
- false triggers: under 3%

Almost all the time is MediaPipe: about 12 ms for FaceMesh, 10 ms for Hands. The classifier takes under 1 ms, so swapping SVM for LSTM barely matters for latency.

## Layout

```
TrinetraAI/
├── config/          settings.yaml, loader.py
├── vision/          capture, face_tracker, gesture_detector, overlay
├── classifier/      gesture_model, trainer, data/
├── control/         actions, action_mapper
├── utils/           metrics
├── tests/
├── Makefile
└── main.py
```

`make install`, `make run`, `make train`, `make test`, `make lint`, `make format` all work.

## Contributing

PRs welcome. Two things I'd really like help with: smoothing the head pose (a Kalman filter would probably kill the jitter), and testing with people who actually have tremors, since that's who this is for.

## Credits

[MediaPipe](https://google.github.io/mediapipe/), [OpenCV](https://opencv.org/), [scikit-learn](https://scikit-learn.org/), [pynput](https://pynput.readthedocs.io/), [PyAutoGUI](https://pyautogui.readthedocs.io/). Head-pose approach adapted from [learnopencv](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/).

MIT licensed.
