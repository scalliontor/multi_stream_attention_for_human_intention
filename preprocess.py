"""This script is for preprocessing something-something-v2 dataset.
The code is largely borrowed from https://github.com/MIT-HAN-LAB/temporal-shift-module
and https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
"""

import os
import sys
import threading
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='prepare something-something-v2 dataset')
    # Updated default paths to match your current directory structure
    current_dir = '/mnt/DA0054DE0054C365/linh_tinh/Share_tech/hand/multi_stream_attention'
    parser.add_argument('--video_root', type=str, default=os.path.join(current_dir, '20bn-something-something-v2'))
    parser.add_argument('--frame_root', type=str, default=os.path.join(current_dir, '20bn-something-something-v2-frames'))
    parser.add_argument('--anno_root', type=str, default=os.path.join(current_dir, 'annotations'))
    parser.add_argument('--num_threads', type=int, default=100)
    parser.add_argument('--decode_video', action='store_true', default=True)
    parser.add_argument('--build_file_list', action='store_true', default=True)
    args = parser.parse_args()

    args.video_root = os.path.expanduser(args.video_root)
    args.frame_root = os.path.expanduser(args.frame_root)
    args.anno_root = os.path.expanduser(args.anno_root)
    return args

def split_func(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract(video, tmpl='%06d.jpg'):
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\" -y -loglevel error'.format(args.video_root, video, args.frame_root, video[:-5])
    result = os.system(cmd)
    return result == 0

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
    
    # Split remaining videos among threads
    videos_per_thread = len(remaining_videos) // args.num_threads
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