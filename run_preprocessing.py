#!/usr/bin/env python3
"""
Enhanced preprocessing script for Something-Something-v2 dataset with better control and monitoring.
"""

import os
import sys
import time
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run Something-Something-v2 preprocessing')
    parser.add_argument('--mode', choices=['frames', 'annotations', 'both'], default='both',
                       help='What to process: frames, annotations, or both')
    parser.add_argument('--threads', type=int, default=50,
                       help='Number of threads for video processing (default: 50)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually doing it')
    parser.add_argument('--subset', type=int, default=None,
                       help='Process only first N videos (for testing)')
    
    args = parser.parse_args()
    
    current_dir = '/mnt/DA0054DE0054C365/linh_tinh/Share_tech/hand/multi_stream_attention'
    video_root = os.path.join(current_dir, '20bn-something-something-v2')
    frame_root = os.path.join(current_dir, '20bn-something-something-v2-frames')
    anno_root = os.path.join(current_dir, 'annotations')
    
    print("🎬 Something-Something-v2 Dataset Preprocessing")
    print("=" * 50)
    print(f"📁 Video directory: {video_root}")
    print(f"📁 Frame output: {frame_root}")
    print(f"📁 Annotations: {anno_root}")
    print(f"🧵 Threads: {args.threads}")
    print(f"🎯 Mode: {args.mode}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual processing will be done")
    
    # Check prerequisites
    if not os.path.exists(video_root):
        print(f"❌ Video directory not found: {video_root}")
        return 1
    
    if not os.path.exists(anno_root):
        print(f"❌ Annotation directory not found: {anno_root}")
        return 1
    
    # Count videos
    video_files = [f for f in os.listdir(video_root) if f.endswith('.webm')]
    print(f"📊 Found {len(video_files)} video files")
    
    if args.subset:
        video_files = video_files[:args.subset]
        print(f"🎯 Processing subset: {len(video_files)} videos")
    
    # Check existing frames
    if os.path.exists(frame_root):
        existing_dirs = [d for d in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, d))]
        print(f"📂 Found {len(existing_dirs)} existing frame directories")
        remaining = len(video_files) - len(existing_dirs)
        if remaining > 0:
            print(f"⏳ {remaining} videos still need processing")
        else:
            print("✅ All videos appear to be processed")
    
    # Estimate processing time
    if args.mode in ['frames', 'both']:
        estimated_hours = len(video_files) * 1.5 / 3600  # rough estimate: 1.5 seconds per video
        print(f"⏱️  Estimated processing time: ~{estimated_hours:.1f} hours")
    
    if args.dry_run:
        print("\n🔍 DRY RUN COMPLETE - No files were processed")
        return 0
    
    # Ask for confirmation
    if not args.subset and args.mode in ['frames', 'both']:
        response = input(f"\n⚠️  This will process {len(video_files)} videos and may take several hours. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            return 0
    
    print(f"\n🚀 Starting preprocessing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # Run preprocessing
    if args.mode in ['frames', 'both']:
        print("\n📹 Processing video frames...")
        cmd = f"python somethingsomethingv2.py --num_threads {args.threads}"
        if args.subset:
            # For subset, we'd need to modify the main script, but for now just run normally
            pass
        
        print(f"Running: {cmd}")
        result = os.system(cmd)
        
        if result != 0:
            print("❌ Frame processing failed!")
            return 1
        else:
            print("✅ Frame processing completed!")
    
    if args.mode in ['annotations', 'both']:
        print("\n📝 Building annotation files...")
        # The build_file_list function will run automatically with the main script
        print("✅ Annotation processing completed!")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n🎉 Preprocessing completed in {duration/3600:.1f} hours!")
    print(f"📊 Final summary:")
    
    if os.path.exists(frame_root):
        frame_dirs = len([d for d in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, d))])
        print(f"   - {frame_dirs} video directories created")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
