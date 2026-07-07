# README (English)

## Overview

This project maps **Apple Vision Pro** hand tracking data to a D1 robot in **Isaac Gym**, with two usage modes:

- `d1_isaac_control.py`: real-time teleoperation
- `d1_isaac_replay.py`: offline replay

Example directory structure:

```text
avp_teleop/
├── d1_isaac_control.py
├── d1_isaac_replay.py
├── demo_data/
├── dex-retargeting/
├── README.md
└── README_中文.md
```

---

## 1. Requirements

- Python **3.8** (recommended)
- Isaac Gym (**Preview 4**)
- Installed and importable:
  - `beingbeyond_d1_sdk`
  - `avp_stream`
  - `pinocchio`
  - `scipy`
  - `opencv-python`
  - `dex_retargeting`

### Important versions (recommended to pin)
- `numpy==1.24.4`
- `pin==2.7.0`

---

## 2. Installation

### Step 1: Enter the project directory
```bash
cd avp_teleop
```

### Step 2: Install key pinned dependencies
```bash
pip install "avp_stream"
pip install "numpy==1.24.4"
pip install "pin==2.7.0"
```

### Step 3: Install local `dex_retargeting`
```bash
cd dex-retargeting
pip install -e .
cd ..
```

Check installation:
```bash
pip show dex_retargeting
```

### Step 4: Install other common dependencies
```bash
pip install scipy opencv-python
```

---

## 3. Run

### A. Real-time Teleoperation (Vision Pro)

Script: `d1_isaac_control.py`

Features:
- A subprocess connects to Vision Pro and writes data into shared memory
- Main process reads shared memory, performs hand retargeting + IK, and drives Isaac Gym
- Optional video feedback from Isaac Gym camera to Vision Pro

Run command:
```bash
python d1_isaac_control.py --ip <VISION_PRO_IP>
```

Example:
```bash
python d1_isaac_control.py --ip 192.168.20.100
```

Notes:
- Without `--ip`, it runs in simulation-only mode (no external input):
  ```bash
  python d1_isaac_control.py
  ```
- After the Isaac Gym viewer opens, press **`S`** to:
  - Calibrate (set the current hand pose as the reference)
  - Start teleoperation

---

### B. Offline Replay (`.npz` data)

Script: `d1_isaac_replay.py`

Run command:
```bash
python d1_isaac_replay.py
```

Notes:
- By default, it loads the `.npz` file specified by `ReplayCfg.tracking_data_path`
- Replay automatically resets and loops when it reaches the end

If you want to use your own data file, edit `d1_isaac_replay.py` and modify:
```python
tracking_data_path = "demo_data/demo_data.npz"
```
---

## 4. `.npz` Data Format (for replay)

Required fields for `d1_isaac_replay.py`:

### Required
- `timestamps`: `(T,)`
- `right_wrist_poses`: `(T, 4, 4)`
- `right_pinch`: `(T,)`

### Optional
- `right_fingers`: `(T, 25, 4, 4)`