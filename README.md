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
- **Dataset Labels**: Download these 4 files from Something-Something-v2:
  - `something-something-v2-train.json`
  - `something-something-v2-validation.json` 
  - `something-something-v2-test.json`
  - `something-something-v2-labels.json`
- **Bounding Box Annotations**: Something-Else (4 parts) from https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ
  - `bounding_box_smthsmth_part1.json`
  gdown --id 1OlggVsZt8eLOI33C3GKAp9EHajMqxQ_Y -O bounding_box_smthsmth_part1.json

  - `bounding_box_smthsmth_part2.json`
  gdown --id 10GQ3RINLAwnw7C2c91Lo17TuD2HnoRgr -O bounding_box_smthsmth_part2.json

  - `bounding_box_smthsmth_part3.json`
  gdown --id 1-kebQmdN4lE6NI3CxGeKLKNJqwaT8uRG -O bounding_box_smthsmth_part3.json

  - `bounding_box_smthsmth_part4.json`
  gdown --id 1oVkc4o8LaWZhF7DLDtqAEVYKOcG0dNjQ -O bounding_box_smthsmth_part4.json


  gdown --id 1WxdhKsftmh5owWsI4_XleDlHP3BG2tvO -O something-something-v2-validation.json

  gdown --id 1w2Vv0RJexb_3m1SdJ20QrkrI4NQH4ZzG -O something-something-v2-train.json

  gdown --id 1hjOyH74BBLwtHwLOQB9H8E5GsyWV7JHE -O something-something-v2-test.json
  
  gdown --id 1gBmFFtxlj8LLVey7w5PPOILvhcGlyI_r -O something-something-v2-labels.json

## 📁 Required Folder Structure

Your project should look like this:
```
multi_stream_attention_for_human_intention/
├── README.md
├── model.py
├── train.py
├── preprocess.py
├── inference.py
├── test.py
├── extract_frames.py
├── merge_annotations.py
├── requirements.txt
├── videos/
│   ├── 20bn-something-something-v2/     # Extracted .webm video files
│   ├── full.zip                         # Original download (can delete after extract)
│   ├── part0.zip                        # Original download (can delete after extract)
│   └── part1.zip                        # Original download (can delete after extract)
├── bounding_box_smthsmth_part1.json     # Something-Else annotations
├── bounding_box_smthsmth_part2.json     # (Download these 4 files)
├── bounding_box_smthsmth_part3.json
├── bounding_box_smthsmth_part4.json
├── something-something-v2-train.json    # Training split
├── something-something-v2-validation.json # Validation split
├── something-something-v2-test.json     # Test split
├── something-something-v2-labels.json   # Action class labels
└── annotations.json                     # Created by merge_annotations.py
```

After processing, you'll also have:
```
├── frames/                              # Created by extract_frames.py
│   ├── video_id_1/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── video_id_2/
└── processed_data/                      # Created by preprocess.py
    ├── video_id_1/
    │   ├── frames/
    │   ├── hand_landmarks/
    │   └── object_crops/
    └── video_id_2/
```

### 3. Process Data
```bash
# Merge annotations (4 parts into 1 file)
python merge_annotations.py --input_dir . --output annotations.json

# Extract frames from .webm videos
python extract_frames.py --video_dir videos/20bn-something-something-v2/ --output_dir frames/

# HYBRID preprocessing (GT ROI + MediaPipe)
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
