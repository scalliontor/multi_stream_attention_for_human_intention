# Multi-Stream Cross-Attention Synthesizer

Video action recognition using **HYBRID preprocessing**: Ground Truth ROI + MediaPipe joints.

## 🏗️ Architecture
1. **Hand Stream**: GT Hand ROI → MediaPipe → GNN (128D)
2. **Object Stream**: GT Object Bbox → MobileNetV2 (256D)  
3. **Context Stream**: Full Frames → ResNet34 (512D)
4. **Fusion**: Cross-Attention → BiLSTM → 174 classes

## 🔬 HYBRID Preprocessing (Best Approach)
- **Ground Truth ROI**: Something-Else hand bboxes guide MediaPipe
- **Rich Joints**: 21 precise joint coordinates (vs 4 bbox numbers)
- **4x Faster**: MediaPipe on small crops vs full frames
- **Higher Accuracy**: Clean hand regions → better joint detection

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Download Data
- **Videos**: Something-Something-v2 from https://20bn.com/datasets/something-something
- **Annotations**: Something-Else (4 parts) from https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ

### 3. Process Data
```bash
# Merge annotations
python merge_annotations.py --input_dir . --output annotations.json

# Extract frames
python extract_frames.py --video_dir videos/ --output_dir frames/

# HYBRID preprocessing  
python preprocess.py --frames_dir frames/ --annotations annotations.json --output processed_data/
```

### 4. Train
```bash
python train.py --data_dir processed_data/ --epochs 50 --batch_size 8
```

## 📊 Expected Results
- **Accuracy**: ~65-70% on Something-Something-v2
- **Training**: ~12 hours for 50 epochs (RTX 3080)
- **Processing**: ~3-4 hours for full dataset

## 🔧 GPU Settings
Edit `train.py` parameters:
```python
BATCH_SIZE = 8      # Reduce if OOM
IMAGE_SIZE = 112    # 224 for better quality
SEQUENCE_LENGTH = 20 # Reduce if OOM
```
