"""
Complete testing pipeline for Multi-Stream Cross-Attention Synthesizer.
This implements the rigorous evaluation methodology described in the plan.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from train import SimpleDataset
from model import create_synthesizer_model
import warnings
warnings.filterwarnings('ignore')

def create_test_dataset(test_frame_root, test_annotation_file, max_samples=None):
    """
    Create test dataset that loads preprocessed data with clip-level heuristic crops.
    
    Args:
        test_frame_root: Path to test frames (with object_crops and hand_landmarks subdirs)
        test_annotation_file: Path to test annotations JSON
        max_samples: Limit test samples (None = use all)
        
    Returns:
        Dataset instance for testing
    """
    print(f"📊 Creating test dataset...")
    print(f"   Frame root: {test_frame_root}")
    print(f"   Annotations: {test_annotation_file}")
    
    test_dataset = SimpleDataset(
        frame_root=test_frame_root,
        annotation_file=test_annotation_file, 
        max_samples=max_samples
    )
    
    print(f"✅ Test dataset created with {len(test_dataset)} samples")
    return test_dataset

def evaluate_model(model, test_loader, device, save_predictions=False, predictions_file='test_predictions.json'):
    """
    Run comprehensive evaluation on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        save_predictions: Whether to save detailed predictions
        predictions_file: File to save predictions to
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("🧪 Starting comprehensive model evaluation...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Tracking variables
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    predictions_list = []
    
    # Top-k accuracy tracking
    top5_correct = 0
    
    # Disable gradient calculations to save memory and speed up
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            hand_landmarks = batch['hand_landmarks'].to(device)
            object_crops = batch['object_crops'].to(device)
            context_frames = batch['context_frames'].to(device)
            labels = batch['label'].to(device)
            
            # Get model predictions
            outputs = model(hand_landmarks, object_crops, context_frames)
            
            # Top-1 predictions
            _, predicted = outputs.max(1)
            
            # Top-5 predictions
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            
            # Update total counts
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            for i in range(labels.size(0)):
                if labels[i] in top5_pred[i]:
                    top5_correct += 1
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0
                
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
            
            # Save detailed predictions if requested
            if save_predictions:
                for i in range(labels.size(0)):
                    pred_data = {
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'confidence': torch.softmax(outputs[i], 0).max().item(),
                        'top5_predictions': top5_pred[i].cpu().tolist(),
                        'top5_confidences': torch.softmax(outputs[i], 0).topk(5)[0].cpu().tolist()
                    }
                    predictions_list.append(pred_data)
            
            # Progress reporting
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches...")
    
    # Calculate metrics
    top1_accuracy = 100 * correct / total
    top5_accuracy = 100 * top5_correct / total
    
    # Per-class accuracy
    per_class_accuracy = {}
    for class_id in class_total:
        if class_total[class_id] > 0:
            per_class_accuracy[class_id] = 100 * class_correct[class_id] / class_total[class_id]
    
    # Calculate mean per-class accuracy
    mean_per_class_accuracy = np.mean(list(per_class_accuracy.values())) if per_class_accuracy else 0.0
    
    # Save predictions if requested
    if save_predictions and predictions_list:
        print(f"💾 Saving detailed predictions to {predictions_file}")
        with open(predictions_file, 'w') as f:
            json.dump(predictions_list, f, indent=2)
    
    # Compile results
    results = {
        'total_samples': total,
        'correct_predictions': correct,
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'mean_per_class_accuracy': mean_per_class_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'num_classes_evaluated': len(per_class_accuracy)
    }
    
    return results

def print_evaluation_results(results):
    """Print comprehensive evaluation results."""
    print("\n" + "="*60)
    print("🎯 EVALUATION RESULTS")
    print("="*60)
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total test samples: {results['total_samples']:,}")
    print(f"   Classes evaluated: {results['num_classes_evaluated']}")
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"   Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    print(f"   Mean Per-Class Accuracy: {results['mean_per_class_accuracy']:.2f}%")
    
    print(f"\n📈 Performance Analysis:")
    if results['top1_accuracy'] > 50:
        print("   ✅ Excellent performance (>50% accuracy)")
    elif results['top1_accuracy'] > 30:
        print("   ✅ Good performance (30-50% accuracy)")
    elif results['top1_accuracy'] > 15:
        print("   ⚠️  Moderate performance (15-30% accuracy)")
    else:
        print("   ❌ Needs improvement (<15% accuracy)")
    
    # Top-5 vs Top-1 analysis
    top5_improvement = results['top5_accuracy'] - results['top1_accuracy']
    print(f"   Top-5 improvement: +{top5_improvement:.2f}%")
    
    if top5_improvement > 20:
        print("   📝 Model shows good confidence calibration")
    else:
        print("   📝 Model may be overconfident")

def run_full_evaluation(model_checkpoint_path, test_frame_root, test_annotation_file, 
                       batch_size=8, num_workers=2, max_test_samples=None, 
                       save_predictions=True):
    """
    Run the complete evaluation pipeline.
    
    Args:
        model_checkpoint_path: Path to trained model checkpoint
        test_frame_root: Path to preprocessed test frames
        test_annotation_file: Path to test annotations
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        max_test_samples: Limit test samples (None = all)
        save_predictions: Whether to save detailed predictions
    """
    print("🚀 Starting Full Evaluation Pipeline")
    print("="*50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create test dataset
    test_dataset = create_test_dataset(
        test_frame_root=test_frame_root,
        test_annotation_file=test_annotation_file,
        max_samples=max_test_samples
    )
    
    # Create test data loader
    print(f"📥 Creating test DataLoader...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle test set
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"✅ Test DataLoader created: {len(test_loader)} batches")
    
    # Load model
    print(f"🤖 Loading model from {model_checkpoint_path}")
    model = create_synthesizer_model(num_classes=174)
    
    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Model checkpoint loaded successfully")
    else:
        print("⚠️  Model checkpoint not found, using untrained model")
    
    model = model.to(device)
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_predictions=save_predictions,
        predictions_file=f'test_predictions_{batch_size}bs.json'
    )
    
    # Print results
    print_evaluation_results(results)
    
    return results

if __name__ == "__main__":
    """
    Example usage:
    python test.py
    """
    
    # Configuration - MODIFY THESE PATHS FOR YOUR SETUP
    MODEL_CHECKPOINT = "best_model.pth"  # Path to your trained model
    TEST_FRAME_ROOT = "20bn-something-something-v2-frames"  # Test frames directory
    TEST_ANNOTATION_FILE = "annotations/something-something-v2-test.json"  # Test annotations
    
    # Evaluation parameters
    BATCH_SIZE = 8
    NUM_WORKERS = 2  
    MAX_TEST_SAMPLES = None  # Use full test set
    SAVE_PREDICTIONS = True
    
    print("🎬 Multi-Stream Cross-Attention Synthesizer - Test Evaluation")
    
    # Check if required files exist
    if not os.path.exists(TEST_ANNOTATION_FILE):
        print(f"❌ Test annotation file not found: {TEST_ANNOTATION_FILE}")
        print("   Please ensure you have the Something-Something-v2 annotations")
        exit(1)
    
    if not os.path.exists(TEST_FRAME_ROOT):
        print(f"❌ Test frame directory not found: {TEST_FRAME_ROOT}")
        print("   Please run preprocessing on test set first")
        exit(1)
    
    # Run evaluation
    try:
        results = run_full_evaluation(
            model_checkpoint_path=MODEL_CHECKPOINT,
            test_frame_root=TEST_FRAME_ROOT,
            test_annotation_file=TEST_ANNOTATION_FILE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            max_test_samples=MAX_TEST_SAMPLES,
            save_predictions=SAVE_PREDICTIONS
        )
        
        print("\n🎉 Evaluation completed successfully!")
        print(f"Final Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
