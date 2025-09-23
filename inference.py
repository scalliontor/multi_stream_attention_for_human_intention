"""
Inference script with OpenCV visualization for Multi-Stream Cross-Attention Synthesizer.
"""

import torch
import torch.nn as nn
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import create_synthesizer_model
from train import SimpleDataset

def visualize_inference(video_frames, hand_landmarks, object_crop, predicted_class, true_class, confidence, idx_to_class):
    """Visualize inference results with OpenCV."""
    
    # Create visualization window
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Multi-Stream Inference Visualization', fontsize=16)
    
    # Show sample frames from the video
    for i, ax in enumerate(axes[0]):
        if i < len(video_frames):
            frame = video_frames[i * (len(video_frames) // 3)]
            # Convert from tensor to numpy and denormalize
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            frame_np = frame_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            frame_np = np.clip(frame_np, 0, 1)
            ax.imshow(frame_np)
            ax.set_title(f'Frame {i * (len(video_frames) // 3)}')
            ax.axis('off')
    
    # Show hand landmarks visualization
    ax = axes[1, 0]
    if hand_landmarks is not None:
        # Plot hand landmarks for first frame
        landmarks = hand_landmarks[0].cpu().numpy()  # [21, 2]
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=50)
        # Connect landmarks to show hand structure
        connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                      (0, 5), (5, 6), (6, 7), (7, 8),   # Index
                      (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                      (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                      (0, 17), (17, 18), (18, 19), (19, 20)]  # Pinky
        for start, end in connections:
            ax.plot([landmarks[start, 0], landmarks[end, 0]], 
                   [landmarks[start, 1], landmarks[end, 1]], 'b-', linewidth=2)
    ax.set_title('Hand Landmarks')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    
    # Show object crop
    ax = axes[1, 1]
    if object_crop is not None:
        crop_np = object_crop.permute(1, 2, 0).cpu().numpy()
        crop_np = crop_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        crop_np = np.clip(crop_np, 0, 1)
        ax.imshow(crop_np)
    ax.set_title('Object Crop')
    ax.axis('off')
    
    # Show prediction results
    ax = axes[1, 2]
    ax.text(0.1, 0.8, f'Predicted: {idx_to_class.get(predicted_class, "Unknown")}', 
            fontsize=12, color='blue', weight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.6, f'True: {idx_to_class.get(true_class, "Unknown")}', 
            fontsize=12, color='green', weight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.4, f'Confidence: {confidence:.4f}', 
            fontsize=12, color='red', weight='bold', transform=ax.transAxes)
    
    status = "✅ CORRECT" if predicted_class == true_class else "❌ INCORRECT"
    color = 'green' if predicted_class == true_class else 'red'
    ax.text(0.1, 0.2, status, fontsize=14, color=color, weight='bold', transform=ax.transAxes)
    ax.set_title('Prediction Results')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Wait for key press to continue
    print("Press any key to continue to next sample...")
    cv2.waitKey(0)

def run_inference_with_visualization():
    """Run inference with OpenCV visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_synthesizer_model(num_classes=174).to(device)
    
    # Try to load trained model
    model_path = 'trained_model.pth'
    if os.path.exists(model_path):
        print(f"✅ Loading trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("⚠️  No trained model found. Using untrained model for demonstration.")
    
    model.eval()
    
    # Load class names
    with open('annotations/something-something-v2-labels.json', 'r') as f:
        labels = json.load(f)
        idx_to_class = {int(v): k for k, v in labels.items()}
    
    # Create dataset for real video data
    val_dataset = SimpleDataset('20bn-something-something-v2-frames', 
                               'annotations/something-something-v2-validation.json', 
                               max_samples=10)  # Just 10 samples for visualization
    
    print(f"Running inference with visualization on {len(val_dataset)} samples...")
    
    for i in range(min(5, len(val_dataset))):  # Show first 5 samples
        sample = val_dataset[i]
        
        hand_landmarks = sample['hand_landmarks'].unsqueeze(0).to(device)
        object_crops = sample['object_crops'].unsqueeze(0).to(device)
        context_frames = sample['context_frames'].unsqueeze(0).to(device)
        true_label = sample['label'].item()
        
        with torch.no_grad():
            outputs = model(hand_landmarks, object_crops, context_frames)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
        
        print(f"\nSample {i+1}:")
        print(f"Predicted: {idx_to_class.get(predicted_class, 'Unknown')}")
        print(f"True: {idx_to_class.get(true_label, 'Unknown')}")
        print(f"Confidence: {confidence:.4f}")
        
        # Visualize
        visualize_inference(
            video_frames=context_frames[0],  # Remove batch dimension
            hand_landmarks=hand_landmarks[0],
            object_crop=object_crops[0, 0],  # First frame of object crops
            predicted_class=predicted_class,
            true_class=true_label,
            confidence=confidence,
            idx_to_class=idx_to_class
        )
    
    cv2.destroyAllWindows()

def run_inference():
    """Run inference and compute basic metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_synthesizer_model(num_classes=174).to(device)
    
    # Try to load trained model
    model_path = 'minimal_trained_model.pth'
    if os.path.exists(model_path):
        print(f"✅ Loading trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("⚠️  No trained model found. Using untrained model for demonstration.")
    
    model.eval()
    
    # Load class names
    with open('annotations/something-something-v2-labels.json', 'r') as f:
        labels = json.load(f)
        idx_to_class = {int(v): k for k, v in labels.items()}
    
    print(f"Model loaded with {len(idx_to_class)} action classes")
    
    # Run inference on test samples
    print("\n🔍 Running inference on test samples...")
    
    correct_predictions = 0
    total_samples = 20
    
    with torch.no_grad():
        for i in range(total_samples):
            # Create dummy test data (in real scenario, this would be actual video data)
            batch_size, seq_len = 1, 10
            hand_landmarks = torch.randn(batch_size, seq_len, 21, 2).to(device)
            object_crops = torch.randn(batch_size, seq_len, 3, 112, 112).to(device)
            context_frames = torch.randn(batch_size, seq_len, 3, 112, 112).to(device)
            
            # Random ground truth for testing
            true_label = torch.randint(0, 174, (1,)).item()
            
            # Forward pass
            outputs = model(hand_landmarks, object_crops, context_frames)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
            
            # Get class names
            predicted_action = idx_to_class.get(predicted_class, 'Unknown')
            true_action = idx_to_class.get(true_label, 'Unknown')
            
            # Check if prediction is correct (random for dummy data)
            is_correct = predicted_class == true_label
            if is_correct:
                correct_predictions += 1
            
            # Print sample results
            if i < 5:  # Show first 5 samples
                status = "✅" if is_correct else "❌"
                print(f"{status} Sample {i+1}:")
                print(f"   Predicted: {predicted_action}")
                print(f"   True: {true_action}")
                print(f"   Confidence: {confidence:.4f}")
                print()
    
    # Calculate metrics
    accuracy = correct_predictions / total_samples
    
    print("📊 Inference Results:")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average confidence: {confidence:.4f}")
    
    print("\n🏗️ Architecture Summary:")
    print("├── Stage 1: Parallel Feature Encoding")
    print("│   ├── HandGNNEncoder: 21 landmarks → 128D")
    print("│   ├── ObjectCNNEncoder: MobileNetV2 → 256D")
    print("│   └── ContextCNNEncoder: ResNet34 → 512D")
    print("├── Stage 2: Cross-Attention Fusion → 512D")
    print("└── Stage 3: BiLSTM + Temporal Attention → 174 classes")
    
    print(f"\n📈 Model Statistics:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: ~{sum(p.numel() for p in model.parameters()) * 4 / (1024*1024):.1f} MB")
    
    return accuracy

def test_single_prediction():
    """Test a single prediction with detailed output."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_synthesizer_model(num_classes=174).to(device)
    model.eval()
    
    print("\n🎯 Single Prediction Test:")
    
    with torch.no_grad():
        # Single sample
        hand_landmarks = torch.randn(1, 10, 21, 2).to(device)
        object_crops = torch.randn(1, 10, 3, 112, 112).to(device)
        context_frames = torch.randn(1, 10, 3, 112, 112).to(device)
        
        outputs = model(hand_landmarks, object_crops, context_frames)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
        
        print("Top 5 predictions:")
        for i in range(5):
            class_idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item()
            print(f"{i+1}. Class {class_idx}: {prob:.4f} ({prob*100:.2f}%)")

if __name__ == "__main__":
    print("🚀 Multi-Stream Cross-Attention Synthesizer - Inference")
    print("=" * 60)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--visualize':
        print("Running inference with visualization...")
        run_inference_with_visualization()
    else:
        print("Running standard inference...")
        accuracy = run_inference()
        test_single_prediction()
        
        print("\n✅ Inference completed successfully!")
        print(f"💡 Tip: Use 'python inference.py --visualize' to see visual results!")
        print(f"Note: Results shown are with dummy data for demonstration.")
        print(f"For real results, replace dummy data with actual video frames,")
        print(f"hand landmarks, and object crops from your dataset.")
