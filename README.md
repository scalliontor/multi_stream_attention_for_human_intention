# Multi-Stream Cross-Attention Synthesizer

Video action recognition for Something-Something-v2 dataset.

## 🏗️ Architecture
1. **Hand landmarks** → HandGNNEncoder (128D)
2. **Object crops** → MobileNetV2 (256D)  
3. **Full frames** → ResNet34 (512D)
4. **Cross-attention fusion** → BiLSTM → 174 classes

## 📁 Core Files
- `model.py` - Complete architecture (30M parameters)
- `train.py` - Training with manual parameters
- `inference.py` - Inference with visualization
- `preprocess.py` - Video preprocessing
- `run_preprocessing.py` - Preprocessing runner

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install matplotlib  # For visualization
```

### 2. Train Model
```bash
python train.py
```

### 3. Inference
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
