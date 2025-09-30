"""
Training script with hardware-optimized configurations.
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from model import create_synthesizer_model
import warnings
warnings.filterwarnings('ignore')

class SimpleDataset(Dataset):
    def __init__(self, frame_root, annotation_file, max_samples=1000, labels_file=None):
        # Load annotations for videos we have
        with open(annotation_file, 'r') as f:
            all_annotations = json.load(f)
        
        available_videos = set(os.listdir(frame_root))
        self.annotations = [s for s in all_annotations if s['id'] in available_videos][:max_samples]
        
        # Load labels file - try different locations
        if labels_file is None:
            # Try common locations
            possible_labels = [
                '20bn-something-something-download-package-labels/labels/something-something-v2-labels.json',
                'something-something-v2-labels.json',
                '/something-something-v2-labels.json',
                'test_labels.json'  # For testing
            ]
            labels_file = None
            for path in possible_labels:
                if os.path.exists(path):
                    labels_file = path
                    break
        
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                labels = json.load(f)
                # Create proper class_to_idx mapping (convert string indices to int)
                self.class_to_idx = {k: int(v) for k, v in labels.items()}
                # Also create idx_to_class for reverse lookup
                self.idx_to_class = {int(v): k for k, v in labels.items()}
        else:
            # Fallback: create a simple mapping
            print("⚠️  No labels file found, creating simple mapping")
            unique_templates = list(set([s.get('template', '') for s in self.annotations]))
            self.class_to_idx = {template: i for i, template in enumerate(unique_templates)}
            self.idx_to_class = {i: template for template, i in self.class_to_idx.items()}
        
        self.frame_root = frame_root
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),  # Match IMAGE_SIZE
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        sample = self.annotations[idx]
        video_id = sample['id']
        label = sample.get('template', '').replace('[', '').replace(']', '')
        class_idx = self.class_to_idx.get(label, 0)
        if isinstance(class_idx, str):
            class_idx = 0  # Default to class 0 if mapping fails
        
        # Load context frames (full frames)
        frame_dir = os.path.join(self.frame_root, video_id)
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])[:30]
        
        context_frames = []
        for f in frame_files:
            try:
                img = cv2.imread(os.path.join(frame_dir, f))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                context_frames.append(self.transform(img))
            except:
                continue
        
        # Load Something-Else annotation-based object crops
        object_crops_dir = os.path.join(frame_dir, 'object_crops')
        object_crops = []
        
        if os.path.exists(object_crops_dir):
            # Load object crops extracted from Something-Else annotations
            crop_files = sorted([f for f in os.listdir(object_crops_dir) if f.endswith('.jpg')])[:30]
            for f in crop_files:
                try:
                    img = cv2.imread(os.path.join(object_crops_dir, f))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    object_crops.append(self.transform(img))
                except:
                    continue
        
        # Pad to sequence length with proper normalized zeros
        zero_frame = torch.zeros(3, 112, 112)
        # Apply ImageNet normalization to zero frame
        zero_frame[0] = (0 - 0.485) / 0.229  # R channel
        zero_frame[1] = (0 - 0.456) / 0.224  # G channel  
        zero_frame[2] = (0 - 0.406) / 0.225  # B channel
        
        # Pad context frames
        while len(context_frames) < 30:
            context_frames.append(zero_frame)
        context_frames = context_frames[:30]
        
        # Pad object crops (fallback to context frames if no crops available)
        if len(object_crops) == 0:
            print(f"⚠️  No object crops found for {video_id}, using context frames")
            object_crops = context_frames.copy()
        else:
            while len(object_crops) < 30:
                object_crops.append(zero_frame)
            object_crops = object_crops[:30]
        
        # Load hand landmarks (derived from Something-Else annotations during preprocessing)
        hand_landmarks_dir = os.path.join(frame_dir, 'hand_landmarks')
        hand_landmarks_sequence = []
        
        if os.path.exists(hand_landmarks_dir):
            # Load annotation-derived landmarks
            landmark_files = sorted([f for f in os.listdir(hand_landmarks_dir) if f.endswith('.npy')])[:30]
            for f in landmark_files:
                try:
                    landmarks = np.load(os.path.join(hand_landmarks_dir, f))
                    # Normalize landmarks to [-1, 1] range based on image size
                    landmarks_norm = landmarks.copy().astype(np.float32)
                    landmarks_norm[:, 0] /= 112  # Normalize x by width
                    landmarks_norm[:, 1] /= 112  # Normalize y by height  
                    landmarks_norm = (landmarks_norm - 0.5) * 2  # Map [0,1] -> [-1,1]
                    hand_landmarks_sequence.append(torch.from_numpy(landmarks_norm))
                except:
                    continue
        
        # Pad hand landmarks sequence
        if len(hand_landmarks_sequence) == 0:
            # Fallback to dummy landmarks if no real ones available
            print(f"⚠️  No hand landmarks found for {video_id}, using dummy data")
            hand_landmarks_sequence = [torch.randn(21, 2) * 0.1 for _ in range(30)]
        else:
            # Pad with zeros if sequence too short
            while len(hand_landmarks_sequence) < 30:
                hand_landmarks_sequence.append(torch.zeros(21, 2))
        
        hand_landmarks_sequence = hand_landmarks_sequence[:30]
        hand_landmarks = torch.stack(hand_landmarks_sequence)
        
        return {
            'hand_landmarks': hand_landmarks,
            'object_crops': torch.stack(object_crops),
            'context_frames': torch.stack(context_frames),
            'label': torch.tensor(class_idx, dtype=torch.long)
        }

def train():
    """Simple training function with manual parameters."""
    
    # 🔧 MANUAL PARAMETERS - MODIFY THESE AS NEEDED
    BATCH_SIZE = 2              # Increase if you have more VRAM
    IMAGE_SIZE = 112            # 112 for 4GB, 224 for 8GB+
    SEQUENCE_LENGTH = 10        # 10 for 4GB, 20 for 8GB, 30 for 12GB+
    LEARNING_RATE = 1e-5        # Reduced to prevent NaN
    NUM_EPOCHS = 1              # Adjust based on your time
    GRADIENT_ACCUM_STEPS = 4    # Effective batch = BATCH_SIZE * GRADIENT_ACCUM_STEPS
    MAX_TRAIN_SAMPLES = 1000    # Limit training samples (None = use all)
    MAX_VAL_SAMPLES = 200       # Limit validation samples
    USE_MIXED_PRECISION = False # Disabled to prevent NaN
    NUM_WORKERS = 2
    GRADIENT_CLIP = 1.0         # Prevent gradient explosion             
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training Multi-Stream Cross-Attention Synthesizer")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUM_STEPS}")
    print(f"Epochs: {NUM_EPOCHS}")
    print()
    
    # Create datasets - use processed_data directory
    train_dataset = SimpleDataset('processed_data', 
                                 'something-something-v2-train.json', 
                                 max_samples=MAX_TRAIN_SAMPLES)
    val_dataset = SimpleDataset('processed_data', 
                               'something-something-v2-validation.json', 
                               max_samples=MAX_VAL_SAMPLES)
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=BATCH_SIZE, 
                             shuffle=True, 
                             num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, 
                           batch_size=BATCH_SIZE, 
                           shuffle=False, 
                           num_workers=NUM_WORKERS)
    
    # Create model
    model = create_synthesizer_model(num_classes=174).to(device)
    
    # Initialize model weights properly to prevent NaN
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    # Apply weight initialization
    model.apply(init_weights)
    
    # Check for NaN in model weights
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"⚠️  NaN detected in {name} weights during initialization!")
            param.data.normal_(0, 0.02)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Mixed precision scaler
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            hand_landmarks = batch['hand_landmarks'].to(device)
            object_crops = batch['object_crops'].to(device)
            context_frames = batch['context_frames'].to(device)
            labels = batch['label'].to(device)
            
            try:
                # Check for NaN in inputs
                if torch.isnan(hand_landmarks).any() or torch.isnan(object_crops).any() or torch.isnan(context_frames).any():
                    print(f"⚠️  NaN detected in input data at batch {batch_idx}, skipping...")
                    continue
                
                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = model(hand_landmarks, object_crops, context_frames)
                        loss = criterion(outputs, labels) / GRADIENT_ACCUM_STEPS
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"⚠️  NaN loss detected at batch {batch_idx}, skipping...")
                        continue
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % GRADIENT_ACCUM_STEPS == 0:
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(hand_landmarks, object_crops, context_frames)
                    loss = criterion(outputs, labels) / GRADIENT_ACCUM_STEPS
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"⚠️  NaN loss detected at batch {batch_idx}, skipping...")
                        continue
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % GRADIENT_ACCUM_STEPS == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * GRADIENT_ACCUM_STEPS
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
                
            except torch.cuda.OutOfMemoryError:
                print("❌ GPU out of memory! Try reducing BATCH_SIZE, IMAGE_SIZE, or SEQUENCE_LENGTH")
                return False
            except Exception as e:
                print(f"❌ Error at batch {batch_idx}: {e}")
                continue
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hand_landmarks = batch['hand_landmarks'].to(device)
                object_crops = batch['object_crops'].to(device)
                context_frames = batch['context_frames'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(hand_landmarks, object_crops, context_frames)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}: Val Acc: {val_acc:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Training completed! Model saved as 'trained_model.pth'")

if __name__ == "__main__":
    train()
