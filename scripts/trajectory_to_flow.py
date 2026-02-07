import json
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import argparse

def to_pixel_coords(x, y, W, H):
    # accepts normalized coords [0..1] or pixel coords
    if 0.0 <= x <= 1.01 and 0.0 <= y <= 1.01:
        return int(round(x * W)), int(round(y * H))
    else:
        return int(round(x)), int(round(y))

def gaussian_mask(H, W, cx, cy, radius):
    Y, X = np.mgrid[:H, :W]
    d2 = (X - cx)**2 + (Y - cy)**2
    sigma = max(1.0, radius / 2.0)
    mask = np.exp(-d2 / (2 * sigma * sigma))
    # threshold to binary-ish region for flow support
    mask_bin = ((mask >= np.exp(-(radius*radius)/(2*sigma*sigma))) * 255).astype(np.uint8)
    return mask.astype(np.float32), mask_bin

def main(plan_path="outputs/trajectory_plan.json", first_frame="inputs/first_frame.png", out_dir="outputs/flows", flow_scale=1.0):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "npy"), exist_ok=True)

    plan = json.load(open(plan_path, "r"))
    frames = plan["frames"]
    # read example image size
    img = cv2.imread(first_frame)
    if img is None:
        raise RuntimeError("Cannot read first_frame image: " + first_frame)
    H, W = img.shape[:2]

    # convert frames to (frame_idx -> (x,y,r))
    pos = {f["frame"]: (float(f["x"]), float(f["y"]), float(f.get("radius", 28))) for f in frames}
    num_frames = plan.get("num_frames", max(pos.keys()) + 1)

    # For each timestep t produce flow t -> t+1
    for t in range(num_frames - 1):
        if t not in pos or (t+1) not in pos:
            # produce zero flow and empty mask
            flow = np.zeros((H, W, 2), dtype=np.float32)
            mask_bin = np.zeros((H, W), dtype=np.uint8)
        else:
            x0,y0,r0 = pos[t]
            x1,y1,r1 = pos[t+1]
            cx0, cy0 = to_pixel_coords(x0, y0, W, H)
            cx1, cy1 = to_pixel_coords(x1, y1, W, H)
            # compute displacement vector in pixels
            dx = float(cx1 - cx0)
            dy = float(cy1 - cy0)
            # create soft support mask around the object using gaussian
            gm, mask_bin = gaussian_mask(H, W, cx0, cy0, max(r0, r1))
            # flow at each pixel in support = [dx, dy] * gm_pixel
            flow = np.zeros((H, W, 2), dtype=np.float32)
            flow[..., 0] = dx * gm * flow_scale    # x displacement
            flow[..., 1] = dy * gm * flow_scale    # y displacement
        
        # Ensure mask_bin is valid numpy array - make a clean copy
        if not isinstance(mask_bin, np.ndarray):
            mask_bin = np.array(mask_bin, dtype=np.uint8)
        mask_bin = np.copy(mask_bin).astype(np.uint8)
        
        # save mask and flow
        np.save(os.path.join(out_dir, "npy", f"flow_{t:04d}.npy"), flow)
        
        # Use PIL instead of cv2 for more robust mask writing
        mask_pil = Image.fromarray(mask_bin, mode='L')
        mask_pil.save(os.path.join(out_dir, "masks", f"mask_{t:04d}.png"))
        
        print(f"Saved flow_{t:04d}.npy and mask_{t:04d}.png")
    print("Done. Flows and masks saved to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="outputs/trajectory_plan.json")
    parser.add_argument("--first-frame", default="inputs/first_frame.png")
    parser.add_argument("--out-dir", default="outputs/flows")
    parser.add_argument("--flow-scale", default=1.0, type=float)
    args = parser.parse_args()
    main(args.plan, args.first_frame, args.out_dir, args.flow_scale)
