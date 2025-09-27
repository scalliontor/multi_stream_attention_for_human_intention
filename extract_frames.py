#!/usr/bin/env python3
"""
Extract frames from Something-Something-v2 videos (.webm format).
"""

import os
import cv2
import argparse
import multiprocessing as mp
from glob import glob
from tqdm import tqdm
from pathlib import Path

def extract_video_frames(video_path, output_dir, fps=12):
    """
    Extract frames from a single video.
    
    Args:
        video_path: Path to .webm video file
        output_dir: Directory to save extracted frames
        fps: Target frame rate (Something-Something uses 12 fps)
    
    Returns:
        (video_id, success, frame_count)
    """
    
    video_id = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_id)
    
    # Skip if already processed
    if os.path.exists(video_output_dir):
        existing_frames = len([f for f in os.listdir(video_output_dir) if f.endswith('.jpg')])
        if existing_frames > 0:
            return video_id, True, existing_frames
    
    # Create output directory
    os.makedirs(video_output_dir, exist_ok=True)
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_id, False, 0
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval
        if original_fps > 0:
            frame_interval = max(1, int(original_fps / fps))
        else:
            frame_interval = 1
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at target fps
            if frame_count % frame_interval == 0:
                frame_filename = f"{saved_count + 1:04d}.jpg"
                frame_path = os.path.join(video_output_dir, frame_filename)
                
                # Save frame
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        return video_id, True, saved_count
        
    except Exception as e:
        return video_id, False, 0

def extract_frames_parallel(video_dir, output_dir, num_workers=4, fps=12):
    """
    Extract frames from all videos in parallel.
    
    Args:
        video_dir: Directory containing .webm video files
        output_dir: Directory to save extracted frames
        num_workers: Number of parallel processes
        fps: Target frame rate
    """
    
    print(f"🎬 Extracting frames from videos in: {video_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🧵 Workers: {num_workers}")
    print(f"📹 Target FPS: {fps}")
    
    # Find all video files
    video_pattern = os.path.join(video_dir, "*.webm")
    video_files = sorted(glob(video_pattern))
    
    if not video_files:
        print(f"❌ No .webm files found in: {video_dir}")
        return False
    
    print(f"📊 Found {len(video_files):,} video files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check existing progress
    existing_dirs = [d for d in os.listdir(output_dir) 
                    if os.path.isdir(os.path.join(output_dir, d))]
    print(f"📂 Found {len(existing_dirs):,} existing frame directories")
    
    # Prepare arguments for parallel processing
    args_list = [(video_path, output_dir, fps) for video_path in video_files]
    
    # Process videos in parallel
    successful = 0
    failed = 0
    total_frames = 0
    
    print(f"\n🚀 Starting frame extraction...")
    
    with mp.Pool(processes=num_workers) as pool:
        results = []
        
        # Submit all jobs
        for args in args_list:
            result = pool.apply_async(extract_video_frames, args)
            results.append(result)
        
        # Collect results with progress bar
        for result in tqdm(results, desc="Extracting frames"):
            video_id, success, frame_count = result.get()
            
            if success:
                successful += 1
                total_frames += frame_count
            else:
                failed += 1
                print(f"❌ Failed to process: {video_id}")
    
    print(f"\n✅ Frame extraction complete!")
    print(f"   📊 Successful: {successful:,} videos")
    print(f"   ❌ Failed: {failed:,} videos")
    print(f"   🎞️  Total frames: {total_frames:,}")
    print(f"   📁 Output: {output_dir}")
    
    return successful > 0

def validate_extracted_frames(output_dir, min_frames=5):
    """
    Validate extracted frames.
    
    Args:
        output_dir: Directory containing extracted frames
        min_frames: Minimum expected frames per video
    """
    
    print(f"\n🔍 Validating extracted frames in: {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return False
    
    video_dirs = [d for d in os.listdir(output_dir) 
                 if os.path.isdir(os.path.join(output_dir, d))]
    
    if not video_dirs:
        print("❌ No video directories found")
        return False
    
    print(f"📊 Checking {len(video_dirs):,} video directories...")
    
    valid_videos = 0
    total_frames = 0
    issues = []
    
    for video_dir in tqdm(video_dirs[:100], desc="Validating"):  # Sample first 100
        video_path = os.path.join(output_dir, video_dir)
        frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
        
        if len(frame_files) >= min_frames:
            valid_videos += 1
            total_frames += len(frame_files)
        else:
            issues.append(f"{video_dir}: {len(frame_files)} frames")
    
    print(f"✅ Validation results:")
    print(f"   📊 Valid videos: {valid_videos:,}/{len(video_dirs[:100]):,}")
    print(f"   🎞️  Average frames per video: {total_frames/max(valid_videos,1):.1f}")
    
    if issues:
        print(f"   ⚠️  Issues found: {len(issues)}")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"      - {issue}")
        if len(issues) > 5:
            print(f"      ... and {len(issues)-5} more")
    
    return valid_videos > 0

def main():
    parser = argparse.ArgumentParser(description='Extract frames from Something-Something-v2 videos')
    parser.add_argument('--video_dir', required=True,
                       help='Directory containing .webm video files')
    parser.add_argument('--output_dir', default='frames',
                       help='Output directory for extracted frames')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--fps', type=int, default=12,
                       help='Target frame rate (default: 12)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate extracted frames after processing')
    
    args = parser.parse_args()
    
    print("🎬 Something-Something-v2 Frame Extractor")
    print("=" * 45)
    
    # Extract frames
    success = extract_frames_parallel(
        args.video_dir, 
        args.output_dir, 
        args.workers, 
        args.fps
    )
    
    if not success:
        print("❌ Frame extraction failed!")
        return 1
    
    # Optional validation
    if args.validate:
        validate_success = validate_extracted_frames(args.output_dir)
        if not validate_success:
            print("❌ Validation failed!")
            return 1
    
    print(f"\n🎉 Success! Frames extracted to: {args.output_dir}")
    print("💡 Ready for HYBRID preprocessing")
    
    return 0

if __name__ == "__main__":
    exit(main())
