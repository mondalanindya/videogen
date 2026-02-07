#!/usr/bin/env python3
"""
Physics-based trajectory simulator for bouncing ball.
Generates deterministic bounce trajectories with explicit keyframe marking.
"""

import math
import json
import argparse
import sys


def simulate_bounces(
    num_frames=160,
    fps=16,
    start_x=256,
    start_y=50,
    table_y=420,
    gravity=980.0,
    restitution=0.65,
    start_vx=0.0,
    start_vy=0.0,
    radius=28,
    width=512,
    height=512,
    contact_threshold=2.0
):
    """
    Simulate a bouncing ball using simple 2D physics.
    
    Args:
        num_frames: number of frames to simulate
        fps: frames per second
        start_x, start_y: initial ball position (pixels)
        table_y: y-coordinate of table surface (pixels, larger = lower)
        gravity: gravitational acceleration (pixels/sec^2)
        restitution: coefficient of restitution (0-1, energy retained after bounce)
        start_vx, start_vy: initial velocity (pixels/sec, negative vy = upward)
        radius: ball radius (pixels)
        width, height: frame dimensions (pixels)
        contact_threshold: distance threshold to detect contact (pixels)
    
    Returns:
        dict with trajectory plan format
    """
    dt = 1.0 / fps
    
    x = start_x
    y = start_y
    vx = start_vx
    vy = start_vy
    
    frames = []
    num_contacts = 0
    last_contact_frame = -10  # Track last contact to avoid duplicates
    
    for i in range(num_frames):
        # Apply gravity
        vy += gravity * dt
        
        # Update position
        x += vx * dt
        y += vy * dt
        
        # Check for table contact (ball center reaches table_y)
        is_contact = False
        if y + radius >= table_y:
            # Contact detected - only mark as keyframe if it's a new bounce
            y = table_y - radius
            
            # Only count as new contact if sufficient velocity and time since last contact
            if abs(vy) > 50.0 and (i - last_contact_frame) > 5:
                is_contact = True
                num_contacts += 1
                last_contact_frame = i
            
            # Apply restitution
            vy = -vy * restitution
            
            # Reduce horizontal velocity slightly due to friction
            vx *= 0.98
            
            # Stop if velocity is very small (ball has settled)
            if abs(vy) < 10.0:
                vy = 0
        
        # Boundary checks
        if x < radius:
            x = radius
            vx = -vx * 0.8
        elif x > width - radius:
            x = width - radius
            vx = -vx * 0.8
        
        if y < radius:
            y = radius
            vy = abs(vy)
        
        # Create frame entry
        frame_data = {
            "frame": i,
            "x": float(round(x, 2)),
            "y": float(round(y, 2)),
            "radius": radius,
            "angle": 0.0,
            "keyframe": is_contact
        }
        
        frames.append(frame_data)
    
    print(f"Simulation complete: {num_contacts} contacts detected", file=sys.stderr)
    
    return {
        "frames": frames,
        "num_frames": len(frames),
        "prompt": f"A red ball bouncing on a wooden table ({num_contacts} bounces)",
        "physics": {
            "gravity": gravity,
            "restitution": restitution,
            "table_y": table_y,
            "num_contacts": num_contacts
        }
    }


def tune_for_bounce_count(target_bounces=5, **kwargs):
    """
    Automatically tune parameters to achieve target bounce count.
    
    Args:
        target_bounces: desired number of bounces
        **kwargs: simulation parameters
    
    Returns:
        trajectory plan with approximately target_bounces contacts
    """
    # Strategy: binary search on initial velocity
    # Higher |start_vy| => more initial energy => more bounces
    
    low_vy = -500
    high_vy = -1500
    best_plan = None
    best_diff = float('inf')
    
    print(f"Tuning for {target_bounces} bounces...", file=sys.stderr)
    
    for iteration in range(20):
        mid_vy = (low_vy + high_vy) / 2.0
        
        params = kwargs.copy()
        params['start_vy'] = mid_vy
        plan = simulate_bounces(**params)
        
        actual_bounces = sum(1 for f in plan['frames'] if f['keyframe'])
        diff = abs(actual_bounces - target_bounces)
        
        print(f"  Iteration {iteration+1}: start_vy={mid_vy:.1f} => {actual_bounces} bounces", 
              file=sys.stderr)
        
        if diff < best_diff:
            best_diff = diff
            best_plan = plan
        
        if actual_bounces == target_bounces:
            return plan
        elif actual_bounces < target_bounces:
            # Need more energy
            high_vy = mid_vy
        else:
            # Too much energy
            low_vy = mid_vy
        
        if abs(high_vy - low_vy) < 10:
            break
    
    print(f"Best result: {sum(1 for f in best_plan['frames'] if f['keyframe'])} bounces", 
          file=sys.stderr)
    return best_plan


def main():
    parser = argparse.ArgumentParser(
        description="Simulate bouncing ball trajectory with physics"
    )
    parser.add_argument("--output", "-o", default="outputs/trajectory_plan.json",
                        help="Output JSON file path")
    parser.add_argument("--num-frames", type=int, default=160,
                        help="Number of frames to simulate")
    parser.add_argument("--fps", type=int, default=16,
                        help="Frames per second")
    parser.add_argument("--target-bounces", type=int, default=5,
                        help="Target number of bounces (will tune automatically)")
    parser.add_argument("--gravity", type=float, default=980.0,
                        help="Gravity in pixels/sec^2")
    parser.add_argument("--restitution", type=float, default=0.65,
                        help="Coefficient of restitution (0-1)")
    parser.add_argument("--table-y", type=int, default=420,
                        help="Table surface y-coordinate")
    parser.add_argument("--start-x", type=int, default=256,
                        help="Starting x position")
    parser.add_argument("--start-y", type=int, default=50,
                        help="Starting y position")
    parser.add_argument("--no-tune", action="store_true",
                        help="Don't auto-tune, use start-vy directly")
    parser.add_argument("--start-vy", type=float, default=-900.0,
                        help="Initial vertical velocity (negative = upward)")
    
    args = parser.parse_args()
    
    sim_params = {
        "num_frames": args.num_frames,
        "fps": args.fps,
        "start_x": args.start_x,
        "start_y": args.start_y,
        "table_y": args.table_y,
        "gravity": args.gravity,
        "restitution": args.restitution,
        "start_vy": args.start_vy,
    }
    
    if args.no_tune:
        plan = simulate_bounces(**sim_params)
    else:
        plan = tune_for_bounce_count(target_bounces=args.target_bounces, **sim_params)
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(plan, f, indent=2)
    
    actual_bounces = sum(1 for f in plan['frames'] if f['keyframe'])
    contact_frames = [f['frame'] for f in plan['frames'] if f['keyframe']]
    
    print(f"\nGenerated trajectory: {args.output}")
    print(f"Total frames: {len(plan['frames'])}")
    print(f"Actual bounces: {actual_bounces}")
    print(f"Contact frames: {contact_frames}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
