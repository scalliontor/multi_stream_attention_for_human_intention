# 🚀 Deployment Checklist

## ✅ Pre-deployment (This Computer)
- [x] Clean codebase (removed test files, dev artifacts)
- [x] HYBRID preprocessing pipeline ready
- [x] Core files organized and documented
- [x] Utility scripts created (merge_annotations.py, extract_frames.py)
- [x] Comprehensive README with deployment instructions

## 📦 Transfer to Training Computer
- [ ] Copy entire project directory
- [ ] Verify all core files present (see list below)
- [ ] Install dependencies: `pip install -r requirements.txt`

## 🔧 Setup on Training Computer

### 1. Download Datasets
- [ ] Download Something-Something-v2 videos (.webm files)
- [ ] Download Something-Else annotations (4 parts from Google Drive)

### 2. Data Pipeline
```bash
# Merge annotation files
python merge_annotations.py --input_dir . --output annotations.json

# Extract video frames
python extract_frames.py --video_dir something_something_v2/ --output_dir frames/

# Run HYBRID preprocessing
python preprocess.py --frames_dir frames/ --annotations annotations.json --output processed_data/
```

### 3. Training
```bash
python train.py --data_dir processed_data/ --epochs 50 --batch_size 8
```

## 📁 Required Core Files

### Essential Files (Must be present)
- [ ] `model.py` - Multi-stream architecture
- [ ] `train.py` - Training pipeline  
- [ ] `preprocess.py` - HYBRID preprocessing
- [ ] `inference.py` - Inference script
- [ ] `requirements.txt` - Dependencies
- [ ] `README.md` - Main documentation

### Utility Scripts
- [ ] `merge_annotations.py` - Merge annotation parts
- [ ] `extract_frames.py` - Extract video frames
- [ ] `test.py` - Model evaluation

### Configuration
- [ ] `.gitignore` - Git ignore rules

## 🔬 HYBRID Preprocessing Details

### What It Does
1. **Loads Ground Truth**: Something-Else bounding boxes for hands and objects
2. **ROI-Guided MediaPipe**: Uses hand bbox to crop region → MediaPipe joints
3. **Object Cropping**: Uses object bbox to extract clean object crops
4. **Coordinate Transform**: Converts MediaPipe coords from crop to full frame

### Expected Output Structure
```
processed_data/
├── video_id_1/
│   ├── frames/                # Original video frames
│   ├── hand_landmarks/        # .npy files with (21, 2) joint coordinates
│   └── object_crops/          # .jpg files resized to 112x112
└── video_id_2/
    ├── frames/
    ├── hand_landmarks/
    └── object_crops/
```

## ⚙️ Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB+ VRAM (RTX 3070 or better)
- **RAM**: 32GB+ system memory
- **Storage**: 500GB+ for dataset and processed data
- **CPU**: 8+ cores for preprocessing

### Recommended
- **GPU**: RTX 4080/4090 for faster training
- **RAM**: 64GB for large batch processing
- **Storage**: 1TB+ SSD for faster I/O

## 🧪 Testing Before Training

### Quick Test (No training)
```bash
# Test with small subset
python preprocess.py --frames_dir frames/ --annotations annotations.json --output test_output/ --max_videos 5

# Verify output structure
ls test_output/*/frames/
ls test_output/*/hand_landmarks/
ls test_output/*/object_crops/
```

### Validate Data
```bash
# Check annotation file
python -c "import json; print(f'Videos: {len(json.load(open(\"annotations.json\")))}')"

# Check processed data
python -c "
import os, numpy as np
dirs = os.listdir('test_output')
if dirs:
    landmarks = np.load(f'test_output/{dirs[0]}/hand_landmarks/0001.npy')
    print(f'Landmark shape: {landmarks.shape}')
"
```

## 📊 Expected Performance

### Preprocessing Speed
- **Frame Extraction**: ~1 hour per 10k videos
- **HYBRID Processing**: ~2 hours per 10k videos (with GPU)
- **Total Pipeline**: ~3-4 hours for full dataset

### Training Performance
- **Training Time**: ~12 hours for 50 epochs (RTX 3080)
- **Memory Usage**: ~6GB VRAM with batch_size=8
- **Expected Accuracy**: 65-70% on Something-Something-v2

## 🐛 Common Issues & Solutions

### MediaPipe Installation
```bash
# If MediaPipe fails
pip install --upgrade pip
pip install mediapipe --no-cache-dir
```

### CUDA Issues
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
```bash
# Reduce batch size in train.py
BATCH_SIZE = 4  # Instead of 8
GRADIENT_ACCUM_STEPS = 2  # Maintain effective batch size
```

### Missing Dependencies
```bash
# Install torch-geometric for GNN
pip install torch-geometric

# Install additional dependencies
pip install tqdm matplotlib scikit-learn tensorboard
```

## 🎯 Success Criteria

### Preprocessing Success
- [ ] All videos processed without errors
- [ ] Hand landmarks have shape (21, 2)
- [ ] Object crops are 112x112 pixels
- [ ] MediaPipe success rate >80%

### Training Success
- [ ] Model loads without errors
- [ ] Training loss decreases steadily
- [ ] Validation accuracy reaches >60%
- [ ] No CUDA out-of-memory errors

### Final Model
- [ ] Model saves successfully
- [ ] Inference runs without errors
- [ ] Predictions are reasonable
- [ ] Performance meets expectations

## 📞 Support

### Documentation
- `README.md` - Main deployment guide
- `README_DEPLOYMENT.md` - Detailed deployment instructions

### Key References
- **Something-Something-v2**: https://20bn.com/datasets/something-something
- **Something-Else**: https://joaanna.github.io/something_else/
- **MediaPipe**: https://google.github.io/mediapipe/solutions/hands

---

**Ready for production deployment! 🚀**

**Total estimated setup time: 4-6 hours**
**Total training time: 12-24 hours**
