#!/usr/bin/env python3
"""
Utility functions for trajectory plan analysis.
Includes bounce/contact detection from trajectory data.
"""

import numpy as np
from scipy.signal import find_peaks


def detect_bounces_from_plan(plan, baseline_y=None, min_prominence=8):
    """
    Detect bounce/contact frames from trajectory plan using peak detection.
    
    Args:
        plan: dict with key "frames": list of frame dicts containing 'frame' and 'y'
        baseline_y: optional, if you know table y (e.g. start y ~ 420)
        min_prominence: minimum prominence for peak detection (default 8)
    
    Returns:
        list of frame indices which correspond to contact events (approx)
    
    Strategy:
        - Find local maxima in y (since larger y => lower on screen i.e., contact with table)
        - Also include the initial contact at frame 0 if present
    """
    ys = np.array([f['y'] for f in plan['frames']], dtype=float)
    frames = [f['frame'] for f in plan['frames']]
    
    # Find peaks (local maxima) in y => likely contacts
    peaks, props = find_peaks(ys, prominence=min_prominence)
    contact_frames = [frames[p] for p in peaks.tolist()]
    
    # Include frame 0 if near baseline or if first frame flagged keyframe (optional)
    if plan['frames'][0].get('keyframe', False) and 0 not in contact_frames:
        contact_frames.insert(0, 0)
    
    return sorted(contact_frames)


def detect_apexes_from_plan(plan, min_prominence=8):
    """
    Detect apex (highest point) frames from trajectory plan.
    
    Args:
        plan: dict with key "frames": list of frame dicts containing 'frame' and 'y'
        min_prominence: minimum prominence for peak detection (default 8)
    
    Returns:
        list of frame indices which correspond to apex events
    
    Note:
        This finds local minima in y (since smaller y => higher on screen)
    """
    ys = np.array([f['y'] for f in plan['frames']], dtype=float)
    frames = [f['frame'] for f in plan['frames']]
    
    # Find peaks in -ys (local minima in y) => apexes
    peaks, props = find_peaks(-ys, prominence=min_prominence)
    apex_frames = [frames[p] for p in peaks.tolist()]
    
    return sorted(apex_frames)


def mark_keyframes_in_plan(plan, keyframe_indices):
    """
    Mark specific frame indices as keyframes in the plan.
    
    Args:
        plan: trajectory plan dict
        keyframe_indices: list of frame indices to mark as keyframes
    
    Returns:
        modified plan with keyframe flags set
    """
    frames_map = {f['frame']: f for f in plan['frames']}
    
    for idx in keyframe_indices:
        if idx in frames_map:
            frames_map[idx]['keyframe'] = True
    
    return plan


def get_keyframe_pairs(plan):
    """
    Extract consecutive keyframe pairs from plan for interpolation.
    
    Args:
        plan: trajectory plan dict with keyframe flags
    
    Returns:
        list of (frame_a, frame_b) tuples for consecutive keyframes
    """
    keyframes = sorted([f['frame'] for f in plan['frames'] if f.get('keyframe', False)])
    
    if len(keyframes) < 2:
        print(f"Warning: Only {len(keyframes)} keyframes found, need at least 2")
        return []
    
    pairs = []
    for i in range(len(keyframes) - 1):
        pairs.append((keyframes[i], keyframes[i + 1]))
    
    return pairs


def compute_bounce_statistics(plan):
    """
    Compute statistics about detected bounces.
    
    Args:
        plan: trajectory plan dict
    
    Returns:
        dict with bounce statistics
    """
    contact_frames = detect_bounces_from_plan(plan)
    apex_frames = detect_apexes_from_plan(plan)
    
    ys = np.array([f['y'] for f in plan['frames']], dtype=float)
    
    # Compute bounce heights (distance from contact to apex)
    bounce_heights = []
    for i in range(len(contact_frames) - 1):
        # Find apex between this contact and next
        c1, c2 = contact_frames[i], contact_frames[i + 1]
        apexes_between = [a for a in apex_frames if c1 < a < c2]
        if apexes_between:
            apex_idx = apexes_between[0]
            frames_map = {f['frame']: f for f in plan['frames']}
            if apex_idx in frames_map and c1 in frames_map:
                height = frames_map[c1]['y'] - frames_map[apex_idx]['y']
                bounce_heights.append(height)
    
    return {
        'num_contacts': len(contact_frames),
        'num_apexes': len(apex_frames),
        'contact_frames': contact_frames,
        'apex_frames': apex_frames,
        'bounce_heights': bounce_heights,
        'avg_bounce_height': np.mean(bounce_heights) if bounce_heights else 0,
    }


if __name__ == "__main__":
    import json
    import sys
    
    # Test with current plan
    plan_path = "outputs/trajectory_plan.json"
    if len(sys.argv) > 1:
        plan_path = sys.argv[1]
    
    try:
        with open(plan_path) as f:
            plan = json.load(f)
        
        print(f"Analyzing trajectory plan: {plan_path}")
        print(f"Total frames: {len(plan['frames'])}")
        print()
        
        stats = compute_bounce_statistics(plan)
        print(f"Detected contacts: {stats['num_contacts']}")
        print(f"Contact frames: {stats['contact_frames']}")
        print(f"Detected apexes: {stats['num_apexes']}")
        print(f"Apex frames: {stats['apex_frames']}")
        print(f"Bounce heights: {[f'{h:.1f}' for h in stats['bounce_heights']]}")
        print(f"Average bounce height: {stats['avg_bounce_height']:.1f} pixels")
        
    except FileNotFoundError:
        print(f"Error: {plan_path} not found")
        sys.exit(1)
