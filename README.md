# Multi-Stream Cross-Attention Synthesizer

**Production-ready** video action recognition for Something-Something-v2 dataset using **HYBRID preprocessing** with Something-Else annotations.

## 🏗️ HYBRID Architecture
1. **Hand Stream**: GT ROI → MediaPipe → GNN HandEncoder (128D) - *guided joint detection*
2. **Object Stream**: GT Bbox → MobileNetV2 (256D) - *ground truth object crops*
3. **Context Stream**: Full Frames → ResNet34 (512D) - *global context*
4. **Fusion**: Cross-Attention → BiLSTM → 174 classes

### 🔬 **HYBRID Hand Processing**
- **Ground Truth ROI**: Something-Else hand bboxes guide MediaPipe to correct region
- **Rich Joints**: MediaPipe extracts 21 precise joint coordinates (42 numbers vs 4 bbox numbers)
- **GNN Structure**: Encodes hand skeleton with message passing between connected joints
- **Pose Discrimination**: Can distinguish 'grasping' vs 'releasing' vs 'pointing' actions
- **Higher Accuracy**: MediaPipe works much better on clean hand crops vs full frames

### 🎯 **Why HYBRID is Superior**
- **Information Richness**: 21 joint coordinates provide structural hand pose information
- **Spatial Guidance**: Ground truth ensures MediaPipe analyzes the correct hand
- **Processing Speed**: MediaPipe on small crops is 4x faster than full frames
- **Temporal Consistency**: Ground truth annotations provide consistent tracking
- **No Detection Errors**: Uses manual annotations instead of error-prone detection

## 📁 Core Files
- `model.py` - Multi-stream architecture (29.9M parameters)
- `train.py` - Training pipeline with HYBRID data loading
- `preprocess.py` - **HYBRID preprocessing** (GT ROI + MediaPipe)
- `inference.py` - Inference with visualization
- `merge_annotations.py` - Merge Something-Else annotation parts
- `extract_frames.py` - Extract frames from .webm videos

## 🚀 Deployment Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Datasets

#### Something-Something-v2 Videos
Download from: https://20bn.com/datasets/something-something

#### Something-Else Annotations (4 parts)
Download from: https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ

### 3. Setup Data Pipeline
```bash
# 1. Merge annotation files
python merge_annotations.py --input_dir . --output annotations.json

# 2. Extract video frames  
python extract_frames.py --video_dir something_something_v2/ --output_dir frames/

# 3. Run HYBRID preprocessing
python preprocess.py --frames_dir frames/ --annotations annotations.json --output processed_data/
```

### 4. Train Model
```bash
python train.py --data_dir processed_data/ --epochs 50 --batch_size 8
```

### 5. Evaluate
```bash
python test.py --model_path trained_model.pth --data_dir processed_data/
```

## 📊 Expected Data Structure
```
project/
├── annotations.json              # Merged Something-Else annotations  
├── frames/                      # Extracted video frames
│   ├── video_id/0001.jpg, 0002.jpg, ...
├── processed_data/              # HYBRID preprocessing output
│   ├── video_id/
│   │   ├── frames/             # Original frames
│   │   ├── hand_landmarks/     # MediaPipe joints (.npy)
│   │   └── object_crops/       # GT object crops (112x112)
└── Core files (model.py, train.py, etc.)
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

