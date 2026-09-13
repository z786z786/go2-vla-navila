#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import rosbag


def analyze_stream(bag: rosbag.Bag, topic: str):
    count = 0
    first = None
    last = None
    prev = None
    max_gap = 0.0
    monotonic = True
    for _, _, t in bag.read_messages(topics=[topic]):
        ts = t.to_sec()
        count += 1
        if first is None:
            first = ts
        if prev is not None:
            gap = ts - prev
            if gap < 0:
                monotonic = False
            max_gap = max(max_gap, max(0.0, gap))
        prev = ts
        last = ts
    duration = 0.0 if first is None or last is None else max(0.0, last - first)
    fps = 0.0 if duration <= 1e-6 or count <= 1 else (count - 1) / duration
    return {
        'message_count': count,
        'first_message_time': first,
        'last_message_time': last,
        'observed_fps': fps,
        'max_interframe_gap_s': max_gap,
        'timestamp_monotonic': monotonic,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--color-topic', default='/camera/color/image_raw')
    parser.add_argument('--depth-topic', default='/camera/depth/image_rect_raw')
    args = parser.parse_args()

    bag_path = Path(args.bag)
    result = {
        'bag_path': str(bag_path),
        'bag_size_bytes': bag_path.stat().st_size if bag_path.exists() else 0,
        'bag_parsed': False,
        'color': {},
        'depth': {},
        'topic_message_count': {},
    }
    try:
        with rosbag.Bag(str(bag_path), 'r') as bag:
            info = bag.get_type_and_topic_info()[1]
            for topic, meta in info.items():
                result['topic_message_count'][topic] = int(meta.message_count)
            result['color'] = analyze_stream(bag, args.color_topic)
            result['depth'] = analyze_stream(bag, args.depth_topic)
            result['bag_parsed'] = True
    except Exception as exc:
        result['error'] = f'{exc.__class__.__name__}: {exc}'

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['bag_parsed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
