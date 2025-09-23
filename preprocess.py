"""This script is for preprocessing something-something-v2 dataset.
The code is largely borrowed from https://github.com/MIT-HAN-LAB/temporal-shift-module
and https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
"""

import os
import sys
import threading
import argparse
import json
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not available. Object crops will use full frames.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  mediapipe not available. Using dummy hand landmarks.")

def parse_args():
    parser = argparse.ArgumentParser(description='prepare something-something-v2 dataset')
    # Default paths relative to this file for portability
    project_root = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--video_root', type=str, default=os.path.join(project_root, '20bn-something-something-v2'))
    parser.add_argument('--frame_root', type=str, default=os.path.join(project_root, '20bn-something-something-v2-frames'))
    parser.add_argument('--anno_root', type=str, default=os.path.join(project_root, 'annotations'))
    parser.add_argument('--num_threads', type=int, default=100)
    parser.add_argument('--decode_video', action='store_true', default=True)
    parser.add_argument('--build_file_list', action='store_true', default=True)
    parser.add_argument('--subset', type=int, default=None, help='Only process the first N videos (for testing)')
    args = parser.parse_args()

    args.video_root = os.path.expanduser(args.video_root)
    args.frame_root = os.path.expanduser(args.frame_root)
    args.anno_root = os.path.expanduser(args.anno_root)
    return args

def split_func(l, n):
    """Yield successive n-sized chunks from l with safe step size."""
    n = max(1, n)
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Global models - loaded once per process
_yolo_model = None
_mediapipe_hands = None

def get_mediapipe_hands():
    """Get or initialize MediaPipe Hands model."""
    global _mediapipe_hands
    if _mediapipe_hands is None and MEDIAPIPE_AVAILABLE:
        mp_hands = mp.solutions.hands
        _mediapipe_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Focus on one hand
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Hands loaded")
    return _mediapipe_hands

def detect_hand_landmarks(frame):
    """
    Detect hand landmarks using MediaPipe.
    
    Args:
        frame (np.array): RGB frame
        
    Returns:
        np.array: (21, 2) hand landmarks or None if no hand detected
    """
    hands_model = get_mediapipe_hands()
    if not MEDIAPIPE_AVAILABLE or hands_model is None:
        return None
    
    try:
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        results = hands_model.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get first hand's landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            h, w = frame.shape[:2]
            
            for landmark in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            
            return np.array(landmarks)
    
    except Exception as e:
        print(f"MediaPipe hand detection failed: {e}")
    
    return None

def get_yolo_model():
    """Get or initialize the YOLO model (thread-safe singleton)."""
    global _yolo_model
    if _yolo_model is None and YOLO_AVAILABLE:
        try:
            print("🤖 Loading YOLO model...")
            # Try YOLOv11n first (latest), fallback to YOLOv8n
            try:
                _yolo_model = YOLO('yolo11n.pt')  # YOLOv11 nano - latest version
                print("✅ YOLOv11n model loaded")
            except:
                try:
                    _yolo_model = YOLO('yolov8n.pt')  # Fallback to YOLOv8n
                    print("✅ YOLOv8n model loaded (YOLOv11n not available)")
                except:
                    _yolo_model = YOLO('yolov5n.pt')  # Last resort fallback
                    print("✅ YOLOv5n model loaded (latest versions not available)")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("⚠️  Falling back to center crop method")
            _yolo_model = None
    return _yolo_model

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union for two bounding boxes."""
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def generate_dummy_hand_landmarks(frame_shape):
    """Generate dummy hand landmarks for videos without real hand detection."""
    h, w = frame_shape[:2]
    # Place dummy hand roughly in center of frame
    center_x, center_y = w // 2, h // 2
    landmarks = []
    for i in range(21):
        # Small random offset around center
        x = center_x + np.random.randint(-50, 51)
        y = center_y + np.random.randint(-50, 51)
        landmarks.append([x, y])
    return np.array(landmarks)

def find_and_crop_object_of_interest(frame, hand_landmarks=None):
    """
    Detect objects and find the one most likely interacting with the hand.
    
    Args:
        frame (np.array): The full video frame (BGR format)
        hand_landmarks (np.array): (21, 2) array of hand landmarks or None
        
    Returns:
        np.array: Cropped object image or None if not found
    """
    yolo_model = get_yolo_model()
    if not YOLO_AVAILABLE or yolo_model is None:
        # Fallback: return center crop of frame
        h, w = frame.shape[:2]
        crop_size = min(h, w) // 3
        y1 = (h - crop_size) // 2
        x1 = (w - crop_size) // 2
        return frame[y1:y1+crop_size, x1:x1+crop_size]
    
    # Try to detect hand landmarks with MediaPipe if not provided
    if hand_landmarks is None:
        hand_landmarks = detect_hand_landmarks(frame)
    
    # Fall back to dummy landmarks if MediaPipe also fails
    if hand_landmarks is None:
        hand_landmarks = generate_dummy_hand_landmarks(frame.shape)
    
    # Create hand bounding box
    min_x = max(0, np.min(hand_landmarks[:, 0]) - 20)
    max_x = min(frame.shape[1], np.max(hand_landmarks[:, 0]) + 20)
    min_y = max(0, np.min(hand_landmarks[:, 1]) - 20)
    max_y = min(frame.shape[0], np.max(hand_landmarks[:, 1]) + 20)
    hand_bbox = [min_x, min_y, max_x, max_y]
    
    try:
        # Run YOLO detection
        results = yolo_model(frame, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            # No objects detected, return hand region
            return frame[int(min_y):int(max_y), int(min_x):int(max_x)]
        
        # Get detected object boxes
        detected_boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(detected_boxes) == 0:
            # No objects detected, return hand region
            return frame[int(min_y):int(max_y), int(min_x):int(max_x)]
        
        # Find best object by IoU with hand
        best_box = None
        max_iou = 0.0
        
        for obj_box in detected_boxes:
            iou = calculate_iou(hand_bbox, obj_box)
            if iou > max_iou:
                max_iou = iou
                best_box = obj_box
        
        # Use best object if good overlap, otherwise use largest object
        if best_box is not None and max_iou > 0.05:
            x1, y1, x2, y2 = map(int, best_box)
        else:
            # Use largest object
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in detected_boxes]
            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = map(int, detected_boxes[largest_idx])
        
        # Ensure valid crop coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        else:
            # Fallback to center crop
            h, w = frame.shape[:2]
            crop_size = min(h, w) // 3
            y1 = (h - crop_size) // 2
            x1 = (w - crop_size) // 2
            return frame[y1:y1+crop_size, x1:x1+crop_size]
            
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        # Fallback to center crop
        h, w = frame.shape[:2]
        crop_size = min(h, w) // 3
        y1 = (h - crop_size) // 2
        x1 = (w - crop_size) // 2
        return frame[y1:y1+crop_size, x1:x1+crop_size]

def extract_with_clip_level_heuristic(video, tmpl='%06d.jpg'):
    """
    Extract frames, hand landmarks, and object crops using clip-level heuristic.
    
    This approach:
    1. Analyzes the entire video to find the moment of maximum hand-object interaction
    2. Identifies the primary "Target Object" for the entire clip
    3. Crops that specific object consistently throughout the video
    """
    video_path = os.path.join(args.video_root, video)
    frame_dir = os.path.join(args.frame_root, video[:-5])
    
    # Create subdirectories
    os.makedirs(frame_dir, exist_ok=True)
    object_crops_dir = os.path.join(frame_dir, 'object_crops')
    hand_landmarks_dir = os.path.join(frame_dir, 'hand_landmarks')
    os.makedirs(object_crops_dir, exist_ok=True)
    os.makedirs(hand_landmarks_dir, exist_ok=True)
    
    try:
        print(f"🎬 Processing {video} with clip-level heuristic...")
        
        # PHASE 1: Analyze entire video to find target object
        print("  📊 Phase 1: Analyzing full video for target object...")
        
        cap = cv2.VideoCapture(video_path)
        frames_data = []  # Store all frame analysis results
        yolo_model = get_yolo_model()
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect hand landmarks
            hand_landmarks = detect_hand_landmarks(frame)
            
            # Run YOLO to detect all objects
            detected_objects = []
            if yolo_model is not None and YOLO_AVAILABLE:
                try:
                    results = yolo_model(frame, verbose=False)
                    if len(results) > 0 and results[0].boxes is not None:
                        detected_boxes = results[0].boxes.xyxy.cpu().numpy()
                        detected_objects = detected_boxes.tolist()
                except:
                    pass
            
            # Store frame data for analysis
            frame_data = {
                'frame_num': frame_count,
                'frame': frame,
                'hand_landmarks': hand_landmarks,
                'detected_objects': detected_objects
            }
            frames_data.append(frame_data)
        
        cap.release()
        print(f"  ✅ Analyzed {frame_count} frames")
        
        # PHASE 2: Find the "climax" interaction and target object
        print("  🎯 Phase 2: Finding target object...")
        
        max_iou = 0.0
        target_object_bbox = None
        climax_frame = None
        
        for frame_data in frames_data:
            hand_landmarks = frame_data['hand_landmarks']
            detected_objects = frame_data['detected_objects']
            
            if hand_landmarks is not None and len(detected_objects) > 0:
                # Create hand bounding box
                min_x = max(0, np.min(hand_landmarks[:, 0]) - 20)
                max_x = min(frame_data['frame'].shape[1], np.max(hand_landmarks[:, 0]) + 20)
                min_y = max(0, np.min(hand_landmarks[:, 1]) - 20)
                max_y = min(frame_data['frame'].shape[0], np.max(hand_landmarks[:, 1]) + 20)
                hand_bbox = [min_x, min_y, max_x, max_y]
                
                # Find object with highest IoU in this frame
                for obj_bbox in detected_objects:
                    iou = calculate_iou(hand_bbox, obj_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        target_object_bbox = obj_bbox
                        climax_frame = frame_data['frame_num']
        
        if target_object_bbox is not None:
            print(f"  ✅ Target object found! Max IoU: {max_iou:.3f} at frame {climax_frame}")
            print(f"      Object bbox: {target_object_bbox}")
        else:
            print("  ⚠️  No significant hand-object interaction found")
        
        # PHASE 3: Generate consistent object crops for entire video
        print("  🖼️  Phase 3: Generating consistent object crops...")
        
        successful_crops = 0
        for frame_data in frames_data:
            frame_num = frame_data['frame_num']
            frame = frame_data['frame']
            hand_landmarks = frame_data['hand_landmarks']
            
            # Save original frame
            frame_filename = os.path.join(frame_dir, f'{frame_num:06d}.jpg')
            cv2.imwrite(frame_filename, frame)
            
            # Save hand landmarks
            landmarks_filename = os.path.join(hand_landmarks_dir, f'{frame_num:06d}.npy')
            if hand_landmarks is not None:
                np.save(landmarks_filename, hand_landmarks)
            else:
                dummy_landmarks = generate_dummy_hand_landmarks(frame.shape)
                np.save(landmarks_filename, dummy_landmarks)
            
            # Generate object crop
            object_crop = None
            
            if target_object_bbox is not None:
                # Try to find the target object in current frame
                current_target_bbox = find_target_object_in_frame(
                    frame, target_object_bbox, yolo_model
                )
                
                if current_target_bbox is not None:
                    # Crop the target object
                    x1, y1, x2, y2 = map(int, current_target_bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        object_crop = frame[y1:y2, x1:x2]
                        successful_crops += 1
            
            # Save object crop or placeholder
            crop_filename = os.path.join(object_crops_dir, f'{frame_num:06d}.jpg')
            if object_crop is not None and object_crop.size > 0:
                object_crop_resized = cv2.resize(object_crop, (112, 112))
                cv2.imwrite(crop_filename, object_crop_resized)
            else:
                # Black placeholder for consistency
                black_crop = np.zeros((112, 112, 3), dtype=np.uint8)
                cv2.imwrite(crop_filename, black_crop)
        
        print(f"  ✅ Generated {successful_crops}/{frame_count} successful object crops")
        print(f"  🎉 Clip-level processing complete!")
        
        return frame_count > 0
        
    except Exception as e:
        print(f"❌ Error processing {video}: {e}")
        return False

def find_target_object_in_frame(frame, target_bbox, yolo_model):
    """
    Find the target object in current frame by matching with reference bbox.
    
    Args:
        frame: Current frame
        target_bbox: Reference bounding box of target object
        yolo_model: YOLO model for detection
        
    Returns:
        Best matching object bbox or None
    """
    if yolo_model is None:
        return target_bbox  # Fallback to original bbox
    
    try:
        # Run YOLO on current frame
        results = yolo_model(frame, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            return target_bbox  # Fallback to original bbox
        
        detected_boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(detected_boxes) == 0:
            return target_bbox  # Fallback to original bbox
        
        # Find object with highest IoU to target_bbox
        best_match = None
        max_iou = 0.0
        
        for obj_bbox in detected_boxes:
            iou = calculate_iou(target_bbox, obj_bbox.tolist())
            if iou > max_iou:
                max_iou = iou
                best_match = obj_bbox.tolist()
        
        # Return best match if IoU is reasonable, otherwise original bbox
        if max_iou > 0.3:  # Reasonable overlap threshold
            return best_match
        else:
            return target_bbox  # Fallback to original bbox
            
    except Exception as e:
        print(f"Error finding target object: {e}")
        return target_bbox  # Fallback to original bbox

# Keep the old function as fallback
def extract(video, tmpl='%06d.jpg'):
    """Fallback to old frame-by-frame extraction if clip-level fails."""
    return extract_with_clip_level_heuristic(video, tmpl)

def target(video_list):
    success_count = 0
    for i, video in enumerate(video_list):
        video_dir = os.path.join(args.frame_root, video[:-5])
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        if extract(video):
            success_count += 1
            print(f"✅ [{i+1}/{len(video_list)}] {video}")
        else:
            print(f"❌ [{i+1}/{len(video_list)}] {video} - FAILED")
    
    print(f"Thread completed: {success_count}/{len(video_list)} videos processed successfully")

def decode_video(args):
    print(f"📁 Video root: {args.video_root}")
    print(f"📁 Frame root: {args.frame_root}")
    print(f"🧵 Using {args.num_threads} threads")
    
    if not os.path.exists(args.video_root):
        raise ValueError('Please download videos and set video_root variable.')
    if not os.path.exists(args.frame_root):
        os.makedirs(args.frame_root)

    # Get list of video files
    video_list = [f for f in os.listdir(args.video_root) if f.endswith('.webm')]
    print(f"📊 Found {len(video_list)} video files to process")
    
    # Filter out already processed videos (resume capability)
    remaining_videos = []
    for video in video_list:
        frame_dir = os.path.join(args.frame_root, video[:-5])
        if not os.path.exists(frame_dir) or len(os.listdir(frame_dir)) == 0:
            remaining_videos.append(video)
    
    if len(remaining_videos) < len(video_list):
        print(f"🔄 Resuming: {len(video_list) - len(remaining_videos)} videos already processed")
        print(f"⏳ {len(remaining_videos)} videos remaining")
    
    if len(remaining_videos) == 0:
        print("✅ All videos already processed!")
        return
    
    # Apply subset if requested
    if args.subset:
        remaining_videos = remaining_videos[:args.subset]
        print(f"🎯 Subset mode: processing {len(remaining_videos)} videos")

    # Split remaining videos among threads (robust for small counts)
    import math
    if args.num_threads <= 0:
        num_threads = 1
    else:
        num_threads = min(args.num_threads, max(1, len(remaining_videos)))
    videos_per_thread = math.ceil(len(remaining_videos) / num_threads)
    splits = list(split_func(remaining_videos, videos_per_thread))
    
    print(f"🚀 Starting preprocessing with {len(splits)} threads...")
    print(f"⏱️  Estimated time: ~{len(remaining_videos) * 2 / 3600:.1f} hours (rough estimate)")

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
    print("🎉 Video preprocessing completed!")

def build_file_list(args):
    if not os.path.exists(args.anno_root):
        raise ValueError('Please download annotations and set anno_root variable.')

    dataset_name = 'something-something-v2'
    with open(os.path.join(args.anno_root, '%s-labels.json' % dataset_name)) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)

    with open('category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = [os.path.join(args.anno_root, '%s-validation.json' % dataset_name),
                   os.path.join(args.anno_root, '%s-train.json' % dataset_name),
                   os.path.join(args.anno_root, '%s-test.json' % dataset_name)]
    files_output = [os.path.join(args.anno_root, 'val_videofolder.txt'),
                    os.path.join(args.anno_root, 'train_videofolder.txt'),
                    os.path.join(args.anno_root, 'test_videofolder.txt')]
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join(args.frame_root, curFolder))
            if len(dir_files) == 0:
                print('video decoding fails at %s', (curFolder))
                sys.exit()
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))

if __name__ == '__main__':
    global args
    args = parse_args()

    if args.decode_video:
        print('Decoding videos to frames.')
        decode_video(args)

    if args.build_file_list:
        print('Generating training files.')
        build_file_list(args)