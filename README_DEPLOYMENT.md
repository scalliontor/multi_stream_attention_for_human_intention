# Multi-Stream Attention Network - Deployment Guide

## 🎯 Overview

This is a **production-ready** Multi-Stream Cross-Attention Synthesizer for video action recognition using the Something-Something-v2 dataset with Something-Else annotations.

### 🏗️ HYBRID Architecture
1. **Hand Stream**: Ground Truth ROI → MediaPipe → GNN HandEncoder (128D)
2. **Object Stream**: Ground Truth Bbox → MobileNetV2 (256D) 
3. **Context Stream**: Full Frames → ResNet34 (512D)
4. **Fusion**: Cross-Attention → BiLSTM → 174 classes

## 🚀 Quick Start (Training Computer)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Datasets

#### Something-Something-v2 Videos
```bash
# Download from: https://20bn.com/datasets/something-something
# Extract to: something_something_v2/
```

#### Something-Else Annotations
```bash
# Download 4 parts from: https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ
# Files: bounding_box_smthsmth_part1.json, part2.json, part3.json, part4.json
```

### 3. Merge Annotations
```bash
python merge_annotations.py
# Creates: annotations.json (consolidated bounding boxes)
```

### 4. Extract Video Frames
```bash
python extract_frames.py --video_dir something_something_v2/ --output_dir frames/
# Creates: frames/video_id/0001.jpg, 0002.jpg, ...
```

### 5. Run HYBRID Preprocessing
```bash
python preprocess.py --frames_dir frames/ --annotations annotations.json --output processed_data/
# Creates: processed_data/video_id/{frames,hand_landmarks,object_crops}/
```

### 6. Train Model
```bash
python train.py --data_dir processed_data/ --epochs 50 --batch_size 8
```

## 📁 Expected Directory Structure

```
project/
├── annotations.json                 # Merged Something-Else annotations
├── frames/                         # Extracted video frames
│   ├── 151201/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── 3201/
├── processed_data/                 # HYBRID preprocessing output
│   ├── 151201/
│   │   ├── frames/                # Original frames
│   │   ├── hand_landmarks/        # MediaPipe joints (.npy)
│   │   └── object_crops/          # GT object crops (112x112)
│   └── 3201/
├── model.py                       # Network architecture
├── train.py                       # Training script
├── preprocess.py                  # HYBRID preprocessing
├── inference.py                   # Inference script
└── requirements.txt               # Dependencies
```

## 🔬 HYBRID Preprocessing Details

### What It Does
1. **Loads Ground Truth**: Something-Else bounding boxes for hands and objects
2. **ROI-Guided MediaPipe**: Uses hand bbox to crop region → MediaPipe joints
3. **Object Cropping**: Uses object bbox to extract clean object crops
4. **Coordinate Transform**: Converts MediaPipe coords from crop to full frame

### Why It's Superior
- **Rich Information**: 21 hand joints (42 numbers) vs 4 bbox numbers
- **Higher Accuracy**: MediaPipe works better on clean hand crops
- **Pose Discrimination**: Can distinguish 'grasping' vs 'releasing' vs 'pointing'
- **Spatial Guidance**: Ground truth ensures correct hand analysis

### Output Format
- **Hand Landmarks**: `.npy` files with shape (21, 2) - MediaPipe joint coordinates
- **Object Crops**: `.jpg` files resized to 112x112 pixels
- **Frames**: Original video frames for context stream

## 🧠 Model Architecture

### HandGNNEncoder
```python
# Uses torch-geometric for hand skeleton structure
# 21 joints connected by hand topology
# Message passing between connected joints
# Outputs 128D hand representation
```

### ObjectCNNEncoder  
```python
# MobileNetV2 backbone on 112x112 object crops
# Outputs 256D object representation
```

### ContextCNNEncoder
```python
# ResNet34 on full video frames
# Outputs 512D context representation  
```

### CrossAttentionFusion
```python
# Multi-head attention between hand, object, context
# BiLSTM for temporal modeling
# Final classification to 174 action classes
```

## ⚙️ Configuration

### Training Parameters (train.py)
```python
BATCH_SIZE = 8              # Adjust based on GPU memory
IMAGE_SIZE = 112            # Fixed for object crops
SEQUENCE_LENGTH = 30        # Frames per video
NUM_EPOCHS = 50             # Training epochs
LEARNING_RATE = 0.001       # Adam optimizer
USE_MIXED_PRECISION = True  # Memory optimization
```

### Hardware Requirements
- **GPU**: 8GB+ VRAM (RTX 3070 or better)
- **RAM**: 32GB+ system memory
- **Storage**: 500GB+ for dataset and processed data
- **CPU**: 8+ cores for preprocessing

## 🔧 Preprocessing Scripts

### merge_annotations.py
```python
# Merges 4 Something-Else annotation parts
# Input: bounding_box_smthsmth_part*.json
# Output: annotations.json
```

### extract_frames.py  
```python
# Extracts frames from .webm videos
# Input: something_something_v2/*.webm
# Output: frames/video_id/*.jpg
```

### preprocess.py (HYBRID)
```python
# Main preprocessing with HYBRID approach
# Input: frames/ + annotations.json
# Output: processed_data/ with hand_landmarks + object_crops
```

## 🧪 Testing (No Training)

### Test Preprocessing Only
```bash
python test_preprocessing.py
# Tests HYBRID pipeline on sample data
# Verifies MediaPipe + annotation integration
```

### Validate Data Structure
```bash
python validate_data.py --data_dir processed_data/
# Checks all required files exist
# Validates hand landmark shapes
# Verifies object crop dimensions
```

## 📊 Expected Performance

### Preprocessing Speed
- **MediaPipe on ROI**: ~50ms per frame (vs 200ms full frame)
- **Object Cropping**: ~5ms per frame
- **Total**: ~1 hour per 1000 videos (with GPU)

### Model Performance
- **Training Time**: ~12 hours for 50 epochs (RTX 3080)
- **Inference Speed**: ~30 FPS on GPU
- **Expected Accuracy**: 65-70% on Something-Something-v2

## 🐛 Troubleshooting

### Common Issues

#### MediaPipe Installation
```bash
# If MediaPipe fails to install
pip install --upgrade pip
pip install mediapipe --no-cache-dir
```

#### CUDA/GPU Issues
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Reinstall with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues
```bash
# Reduce batch size in train.py
BATCH_SIZE = 4  # Instead of 8
# Enable gradient accumulation
GRADIENT_ACCUM_STEPS = 2
```

#### Missing Annotations
```bash
# Verify annotation file structure
python -c "import json; print(len(json.load(open('annotations.json'))))"
# Should show ~180k videos
```

## 📈 Monitoring Training

### TensorBoard Logs
```bash
tensorboard --logdir runs/
# View at http://localhost:6006
```

### Key Metrics to Watch
- **Training Loss**: Should decrease steadily
- **Validation Accuracy**: Should increase to 60%+
- **Hand Landmark Quality**: MediaPipe success rate >80%
- **GPU Utilization**: Should be >90% during training

## 🎯 Production Deployment

### Model Export
```python
# Save trained model
torch.save(model.state_dict(), 'final_model.pth')

# Export for inference
torch.jit.script(model).save('model_scripted.pt')
```

### Inference Pipeline
```python
# Load model
model = create_synthesizer_model(num_classes=174)
model.load_state_dict(torch.load('final_model.pth'))

# Process video
frames, landmarks, crops = preprocess_video(video_path)
prediction = model(landmarks, crops, frames)
action_class = prediction.argmax().item()
```

## 📚 References

- **Something-Something-v2**: https://20bn.com/datasets/something-something
- **Something-Else**: https://joaanna.github.io/something_else/
- **MediaPipe Hands**: https://google.github.io/mediapipe/solutions/hands
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

## 🆘 Support

For issues with:
- **Dataset**: Check Something-Something-v2 and Something-Else documentation
- **MediaPipe**: Check Google MediaPipe documentation
- **Training**: Monitor GPU memory and reduce batch size if needed
- **Preprocessing**: Verify annotation file format and video frame extraction

---

**Ready for production training! 🚀**
