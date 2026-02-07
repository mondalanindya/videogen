#!/usr/bin/env python3
"""
Overlay planned trajectory coordinates onto keyframes to visualize coordinate alignment.
"""
import json
import cv2
import glob
import os

plan = json.load(open("outputs/trajectory_plan.json"))
frame_map = {f['frame']: (f['x'], f['y'], f.get('radius', 28)) for f in plan['frames']}
kf_paths = sorted(glob.glob("outputs/keyframes/frame_*.png"))
out_dir = "outputs/debug_overlays"
os.makedirs(out_dir, exist_ok=True)

print(f"Processing {len(kf_paths)} keyframes...")
print(f"Frame map has {len(frame_map)} entries")

for p in kf_paths:
    idx = int(os.path.basename(p).split('_')[-1].split('.')[0])
    im = cv2.imread(p)
    h, w = im.shape[:2]
    
    if idx in frame_map:
        x, y, r = frame_map[idx]
        
        # Handle normalized coords (0..1) vs absolute pixel coords
        if 0.0 <= x <= 1.01 and 0.0 <= y <= 1.01:
            cx = int(round(x * w))
            cy = int(round(y * h))
            coord_type = "normalized"
        else:
            cx = int(round(x))
            cy = int(round(y))
            coord_type = "absolute"
        
        # Draw planned position as green circle
        cv2.circle(im, (cx, cy), int(round(r)), (0, 255, 0), 2)
        
        # Add text labels
        cv2.putText(im, f"plan:({cx},{cy})", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(im, f"raw:({x:.1f},{y:.1f}) {coord_type}", (5, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw crosshair at center
        cv2.line(im, (cx-10, cy), (cx+10, cy), (0, 255, 0), 1)
        cv2.line(im, (cx, cy-10), (cx, cy+10), (0, 255, 0), 1)
        
        print(f"Frame {idx:04d}: raw=({x:.1f}, {y:.1f}) -> pixel=({cx}, {cy}) [{coord_type}]")
    else:
        print(f"Frame {idx:04d}: not in trajectory plan")
    
    cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), im)

print(f"\n✓ Wrote overlays to {out_dir}")
print("Inspect these images - the green circle should align with the red ball")
