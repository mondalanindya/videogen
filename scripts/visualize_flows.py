#!/usr/bin/env python3
"""
Visualize optical flows as arrow fields.
Helps diagnose whether flows point in the correct direction and magnitude.
"""
import numpy as np
import cv2
import glob
import os
from pathlib import Path

def draw_flow(flow):
    """Draw flow field as arrows on a gray background."""
    h, w, _ = flow.shape
    step = max(4, min(h, w) // 64)
    img = np.zeros((h, w, 3), dtype=np.uint8) + 128
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]
            x2 = int(round(x + dx))
            y2 = int(round(y + dy))
            cv2.arrowedLine(img, (x, y), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    
    return img

flows = sorted(glob.glob("outputs/flows/npy/flow_*.npy"))
out = "outputs/flows/visual"
os.makedirs(out, exist_ok=True)

print(f"Found {len(flows)} flow files")
for i, p in enumerate(flows[:20]):
    f = np.load(p)
    vis = draw_flow(f)
    name = Path(p).stem + ".png"
    cv2.imwrite(os.path.join(out, name), vis)
    print(f"[{i+1}/20] Wrote {os.path.join(out, name)}")

print(f"\nFlow visualizations saved to {out}/")
print("Check outputs/flows/visual/flow_0000.png, flow_0001.png, etc.")
print("During descent: arrows should point downward")
print("At apex: arrows should point upward or minimal")
