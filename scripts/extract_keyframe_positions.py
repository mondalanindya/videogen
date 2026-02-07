#!/usr/bin/env python3
"""Extract keyframe positions from trajectory_plan.json for generate_ip_keyframes.py"""

import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, help="Path to trajectory_plan.json")
    parser.add_argument("--keyframe-interval", type=int, default=10, help="Generate keyframe every N frames")
    parser.add_argument("--format", choices=["semicolon", "file"], default="semicolon",
                       help="Output format: 'semicolon' for x:y:r;x:y:r or 'file' for one per line")
    parser.add_argument("--output", help="Output file (for 'file' format)")
    args = parser.parse_args()
    
    with open(args.trajectory) as f:
        plan = json.load(f)
    
    frames = plan['frames']
    
    # Extract keyframes
    positions = []
    for i, frame in enumerate(frames):
        if i % args.keyframe_interval == 0 or frame.get('keyframe', False):
            x = int(frame['x'])
            y = int(frame['y'])
            r = int(frame.get('radius', 28))
            positions.append((x, y, r))
    
    if args.format == "semicolon":
        # Format for command line: "x1:y1:r1;x2:y2:r2;..."
        result = ";".join(f"{x}:{y}:{r}" for x, y, r in positions)
        print(result)
    elif args.format == "file":
        # Format for file: one line per position "x,y,r"
        lines = [f"{x},{y},{r}" for x, y, r in positions]
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            print(f"Wrote {len(lines)} positions to {args.output}")
        else:
            print('\n'.join(lines))

if __name__ == "__main__":
    main()
