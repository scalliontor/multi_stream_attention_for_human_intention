# Multi-Stream Cross-Attention Synthesizer

Video action recognition for Something-Something-v2 dataset.

## 🏗️ Architecture
1. **Hand landmarks** → **MediaPipe + GNN** HandGNNEncoder (128D) - *real hand detection*
2. **Object crops** → **YOLO + MobileNetV2** (256D) - *detects manipulated objects*
3. **Full frames** → ResNet34 (512D)
4. **Cross-attention fusion** → BiLSTM → 174 classes

### 🔬 **GNN Hand Encoder**
- **Real Hand Detection**: MediaPipe detects 21 hand landmarks per frame
- **Structural Knowledge**: Encodes the hand skeleton structure with GNN
- **Message Passing**: Each landmark learns from its connected neighbors (thumb tip ↔ thumb knuckle)
- **Better Generalization**: Understands physical hand constraints
- **Fallback**: Automatically uses MLP if `torch_geometric` not available

### 🎯 **Clip-Level Heuristic Object Detection**
- **3-Phase Processing**: Analyze entire video → Find target object → Generate consistent crops
- **Temporal Consistency**: Tracks the SAME primary object throughout entire video
- **Peak Interaction Detection**: Finds moment of maximum hand-object IoU across all frames
- **MediaPipe Integration**: Uses real hand landmarks to create hand bounding box
- **YOLOv11n Detection**: Detects all objects, selects best hand-object interaction
- **Robust Fallbacks**: Object tracking → center crop → dummy data as needed

### 🔬 **Why Clip-Level Heuristic?**
- **Problem**: Something-Something-v2 has NO bounding box annotations
- **Solution**: Automatically detect the primary manipulated object per video
- **Advantage**: Temporally consistent object crops (not random per-frame crops)
- **Result**: Model learns focused object representations for each action

## 📁 Core Files
- `model.py` - Complete architecture (30M parameters)
- `train.py` - Training with manual parameters
- `test.py` - **Comprehensive test evaluation pipeline**
- `inference.py` - Inference with visualization
- `preprocess.py` - **Clip-level heuristic preprocessing**
- `run_preprocessing.py` - Preprocessing runner

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install matplotlib  # For visualization
```

**For GNN Hand Encoder** (recommended):
```bash
pip install torch-geometric
```

**For YOLO Object Detection** (recommended):
```bash
pip install ultralytics
```

**For MediaPipe Hand Detection** (recommended):
```bash
pip install mediapipe
```
*Note: All have automatic fallbacks if not installed*

### 2. Preprocess with Clip-Level Heuristic
```bash
python run_preprocessing.py --mode both --threads 16
# Creates: frames + object_crops + hand_landmarks for each video
```

### 3. Train Model
```bash
python train.py
```

### 4. Test Model
```bash
python test.py  # Comprehensive evaluation on test set
```

### 5. Inference
```bash
python inference.py                # Standard metrics
python inference.py --visualize    # With visualization
```

## 🔧 Manual Parameters

Edit the parameters at the top of `train.py`:

```python
# 🔧 MANUAL PARAMETERS - MODIFY THESE AS NEEDED
BATCH_SIZE = 2              # Increase if you have more VRAM
IMAGE_SIZE = 112            # 112 for 4GB, 224 for 8GB+
SEQUENCE_LENGTH = 10        # 10 for 4GB, 20 for 8GB, 30 for 12GB+
NUM_EPOCHS = 20             # Adjust based on your time
GRADIENT_ACCUM_STEPS = 4    # Effective batch = BATCH_SIZE * 4
MAX_TRAIN_SAMPLES = 1000    # Limit training samples
USE_MIXED_PRECISION = True  # Saves memory
```

## 💡 Hardware Tips

### Current 4GB GPU (GTX 1650)
- Start with default parameters
- If OOM error: reduce `BATCH_SIZE` to 1
- Expected training time: ~12-24 hours

### Better GPU (8GB+)
- Increase `BATCH_SIZE` to 8
- Increase `IMAGE_SIZE` to 224
- Increase `SEQUENCE_LENGTH` to 20-30
- Reduce `GRADIENT_ACCUM_STEPS` to 2

## 📊 Status
- **Processed Videos**: 63,810/220,847 (28.9%)
- **Ready for training**: ✅

## 🖥️ Processing on a New Computer

Follow these steps to prepare the dataset on a fresh machine.

### 1) Install system dependencies
- Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```
- Verify ffmpeg:
```bash
ffmpeg -version
```

### 2) Prepare directories
- Place the dataset videos (Something-Something-v2 .webm files) under a folder, e.g.:
  - `/data/ssv2/videos` (contains many `.webm` files)
- Ensure annotations JSONs are available under a folder, e.g.:
  - `/data/ssv2/annotations` (contains `something-something-v2-*.json` and `something-something-v2-labels.json`)
- Choose an output folder for extracted frames, e.g.:
  - `/data/ssv2/frames`

If you keep the default project layout, you can just drop them into:
- `20bn-something-something-v2/` (videos)
- `annotations/` (annotation JSONs)
- `20bn-something-something-v2-frames/` will be created automatically

If your labels are in the downloaded folder `20bn-something-something-download-package-labels/`, either rename it to `annotations/` or create a symlink:

```bash
ln -s 20bn-something-something-download-package-labels annotations
```

This lets training/inference find JSONs at `annotations/something-something-v2-*.json`.

### 3) Run preprocessing (local project folder)

Run these from the repository root (this folder):

- Dry run (prints plan only):
```bash
python run_preprocessing.py --mode both --dry-run
```

- Process frames + build file lists (labels under local `annotations/`):
```bash
python run_preprocessing.py --mode both --threads 16
```

- If your labels are under `20bn-something-something-download-package-labels/` and you do NOT want a symlink:
```bash
python run_preprocessing.py --mode both --threads 16 \
  --anno_root ./20bn-something-something-download-package-labels
```

- Process only frames (no annotation rewrite):
```bash
python run_preprocessing.py --mode frames --threads 16
```

- Build annotation file lists only (after frames exist):
```bash
python run_preprocessing.py --mode annotations
```

### 3b) Run preprocessing (portable, with custom absolute paths)
- Dry run (prints plan only):
```bash
python run_preprocessing.py --mode both --dry-run \
  --video_root /data/ssv2/videos \
  --frame_root /data/ssv2/frames \
  --anno_root /data/ssv2/annotations
```

- Process frames + build file lists (custom paths):
```bash
python run_preprocessing.py --mode both --threads 16 \
  --video_root /data/ssv2/videos \
  --frame_root /data/ssv2/frames \
  --anno_root /data/ssv2/annotations
```

- Process only frames (no annotation rewrite):
```bash
python run_preprocessing.py --mode frames --threads 16 \
  --video_root /data/ssv2/videos \
  --frame_root /data/ssv2/frames
```

- Build annotation file lists only (after frames exist):
```bash
python run_preprocessing.py --mode annotations \
  --anno_root /data/ssv2/annotations \
  --frame_root /data/ssv2/frames
```

- Process a small subset (sanity check):
```bash
python run_preprocessing.py --mode both --threads 8 --subset 100 \
  --video_root /data/ssv2/videos \
  --frame_root /data/ssv2/frames \
  --anno_root /data/ssv2/annotations
```

### 4) Notes and tips
- Resumable: the pipeline skips videos that already have extracted frames.
- Threads: set `--threads` based on CPU cores and disk speed (8–32 is typical).
- Disk space: frames take significant space. Expect 150–250 GB for full SSV2.
- Time estimate: shown before start; actual speed depends on CPU/SSD.
- Portability: both `run_preprocessing.py` and `preprocess.py` accept custom paths and default to the project folder if not provided.

### 5) After preprocessing
- Training will look for frames and annotations in the locations you set.
- If you used custom paths, either keep them the same or move/link them to:
  - `20bn-something-something-v2-frames/`
  - `annotations/`

### 6) Full processing recommendations
- Use SSD/NVMe for `--frame_root` to maximize throughput.
- Use 16–32 threads on a modern 8–16 core CPU.
- Run overnight; the job can be stopped and resumed.
- Keep terminal session persistent (tmux/screen) when running long jobs.

