#!/usr/bin/env python3
"""
Merge Something-Else annotation files from multiple parts into a single file.
"""

import os
import json
import argparse
from glob import glob
from tqdm import tqdm

def merge_annotation_files(input_dir=".", output_file="annotations.json"):
    """
    Merge all Something-Else annotation parts into a single file.
    
    Args:
        input_dir: Directory containing bounding_box_smthsmth_part*.json files
        output_file: Output merged annotation file
    """
    
    print("🔗 Merging Something-Else annotation files...")
    
    # Find all bounding box annotation files
    pattern = os.path.join(input_dir, "bounding_box_smthsmth_part*.json")
    bbox_files = sorted(glob(pattern))
    
    if not bbox_files:
        print(f"❌ No annotation files found matching: {pattern}")
        print("💡 Expected files: bounding_box_smthsmth_part1.json, part2.json, etc.")
        return False
    
    print(f"📁 Found {len(bbox_files)} annotation parts:")
    for f in bbox_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"   - {os.path.basename(f)} ({size_mb:.1f} MB)")
    
    # Merge all annotation files
    all_annotations = {}
    total_videos = 0
    
    for bbox_file in tqdm(bbox_files, desc="Merging annotations"):
        print(f"\n📄 Loading {os.path.basename(bbox_file)}...")
        
        try:
            with open(bbox_file, 'r') as f:
                part_data = json.load(f)
            
            video_count = len(part_data)
            print(f"   📊 {video_count:,} videos in this part")
            
            # Check for overlaps
            overlaps = set(all_annotations.keys()) & set(part_data.keys())
            if overlaps:
                print(f"   ⚠️  {len(overlaps)} overlapping video IDs (will overwrite)")
            
            # Merge with main dictionary
            all_annotations.update(part_data)
            total_videos += video_count
            
        except Exception as e:
            print(f"❌ Error loading {bbox_file}: {e}")
            return False
    
    print(f"\n✅ Merged {total_videos:,} total video annotations")
    print(f"📊 Unique videos: {len(all_annotations):,}")
    
    # Save consolidated annotations
    print(f"💾 Saving consolidated annotations to {output_file}...")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(all_annotations, f)
        
        # Verify saved file
        output_size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"✅ Saved {len(all_annotations):,} video annotations ({output_size_mb:.1f} MB)")
        
        # Show sample structure
        sample_video_id = list(all_annotations.keys())[0]
        sample_frames = len(all_annotations[sample_video_id])
        print(f"📋 Sample: Video {sample_video_id} has {sample_frames} frames")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving {output_file}: {e}")
        return False

def validate_annotations(annotation_file):
    """Validate the merged annotation file."""
    
    print(f"\n🔍 Validating {annotation_file}...")
    
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"✅ File loaded successfully")
        print(f"📊 Total videos: {len(annotations):,}")
        
        # Sample validation
        sample_count = min(5, len(annotations))
        sample_videos = list(annotations.keys())[:sample_count]
        
        total_frames = 0
        hand_count = 0
        object_count = 0
        
        for video_id in sample_videos:
            frames = annotations[video_id]
            total_frames += len(frames)
            
            for frame in frames:
                for label in frame.get('labels', []):
                    if label.get('category') == 'hand':
                        hand_count += 1
                    else:
                        object_count += 1
        
        print(f"📋 Sample validation ({sample_count} videos):")
        print(f"   - Total frames: {total_frames:,}")
        print(f"   - Hand annotations: {hand_count:,}")
        print(f"   - Object annotations: {object_count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Merge Something-Else annotation files')
    parser.add_argument('--input_dir', default='.', 
                       help='Directory containing annotation part files')
    parser.add_argument('--output', default='annotations.json',
                       help='Output merged annotation file')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the merged file')
    
    args = parser.parse_args()
    
    print("🔗 Something-Else Annotation Merger")
    print("=" * 40)
    
    # Merge annotations
    success = merge_annotation_files(args.input_dir, args.output)
    
    if not success:
        print("❌ Merge failed!")
        return 1
    
    # Optional validation
    if args.validate:
        validate_success = validate_annotations(args.output)
        if not validate_success:
            print("❌ Validation failed!")
            return 1
    
    print(f"\n🎉 Success! Merged annotations saved to: {args.output}")
    print("💡 Ready for preprocessing with HYBRID approach")
    
    return 0

if __name__ == "__main__":
    exit(main())
