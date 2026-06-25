import os
import sys
# Allow imports from the parent directory (project root) when running this script directly or as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from LLG_solver import init_SkX, init_SC, init_SP, relax_phase_numba, analyze_state

def plot_quiver_frame(spins, ax_val, ay_val, title, outpath):
    fig, ax_plot = plt.subplots(figsize=(8,8))
    
    tiles_x, tiles_y = 1, 1
    tiled_spins = np.tile(spins, (tiles_x, tiles_y, 1))
    L_x, L_y = tiled_spins.shape[0], tiled_spins.shape[1]
    
    X_base, Y_base = np.meshgrid(np.arange(L_x), np.arange(L_y))
    X, Y = X_base * ax_val, Y_base * ay_val
    
    U = tiled_spins[:, :, 0].T
    V = tiled_spins[:, :, 1].T
    Sz = tiled_spins[:, :, 2].T
    
    q = ax_plot.quiver(X, Y, U, V, Sz, cmap="coolwarm", pivot='mid', scale=max(L_x, L_y)*0.8, width=0.005)
    q.set_clim(-1, 1)
    
    rect = plt.Rectangle(((-0.5)*ax_val, (-0.5)*ay_val), L_x*ax_val, L_y*ay_val, 
                         fill=False, edgecolor='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_plot.add_patch(rect)
            
    ax_plot.set_xlim(-ax_val, L_x * ax_val)
    ax_plot.set_ylim(-ay_val, L_y * ay_val)
    ax_plot.set_aspect('equal')
    ax_plot.set_title(title)
    
    # Use subplots_adjust instead of tight_layout to keep the plot frame and box completely rigid
    # and prevent any layout glitching/jumping when tick labels or titles change.
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

def make_video(phase, L, H_scaled, A_scaled, steps_total, chunk_size, fps=10, mode="quiver"):
    if steps_total is None:
        steps_total = int(50000 * (L / 32)**2)
        
    print(f"Generating {mode.upper()} video for {phase} at H={H_scaled}, A={A_scaled} (Max Steps: {steps_total})")
    
    if phase == "SkX":
        spins, ax_val, ay_val = init_SkX(L)
    elif phase == "SC":
        spins, ax_val, ay_val = init_SC(L)
    elif phase == "SP":
        spins, ax_val, ay_val = init_SP(L)
    else:
        raise ValueError("Invalid phase. Choose from SkX, SC, SP.")
        
    os.makedirs("output/videos/llg", exist_ok=True)
    frames = []
    
    prev_f = 0.0
    tol = 1e-8
    
    steps_done = 0
    frame_idx = 0
    
    # Save frame 0 (Initial State)
    frame_path = f"output/videos/llg/temp_frame_{frame_idx}.png"
    if mode == "fft":
        analyze_state(spins, ax_val, ay_val, phase_name=f"{phase} Step {steps_done}", plot_fft=True, outpath_override=frame_path)
    else:
        plot_quiver_frame(spins, ax_val, ay_val, f"{phase} Step {steps_done}", frame_path)
        
    frames.append(imageio.imread(frame_path))
    os.remove(frame_path)
    print(f"Captured initial frame 0.")
    
    while steps_done < steps_total:
        spins, f_tot, ax_val, ay_val, steps_taken, target_ax, target_ay = relax_phase_numba(
            spins, L, H_scaled, A_scaled, chunk_size, tol, ax_val, ay_val, prev_f, 0.05, 0.25, global_step_start=steps_done, iso_scale=False
        )
        
        steps_done += (steps_taken + 1)
        prev_f = f_tot
        frame_idx += 1
        
        frame_path = f"output/videos/llg/temp_frame_{frame_idx}.png"
        if mode == "fft":
            analyze_state(spins, ax_val, ay_val, phase_name=f"{phase} Step {steps_done}", plot_fft=True, outpath_override=frame_path)
        else:
            plot_quiver_frame(spins, ax_val, ay_val, f"{phase} Step {steps_done}", frame_path)
            
        frames.append(imageio.imread(frame_path))
        os.remove(frame_path)
        
        print(f"Step {steps_done}/{steps_total} | Energy: {f_tot:.5f} | ax={ax_val:.4f}, ay={ay_val:.4f}")
        
        if steps_taken + 1 < chunk_size:
            print("Simulation converged early!")
            break
            
    out_video_mp4 = f"output/videos/llg/{phase}_H{H_scaled}_A{A_scaled}_L{L}_{mode}.mp4"
    out_video_gif = f"output/videos/llg/{phase}_H{H_scaled}_A{A_scaled}_L{L}_{mode}.gif"
    
    print("\nStitching frames into video...")
    try:
        imageio.mimsave(out_video_mp4, frames, fps=fps, macro_block_size=None)
        print(f"Video saved successfully to: {out_video_mp4}")
    except Exception as e:
        print(f"Could not save MP4 (possibly missing ffmpeg dependencies). Error: {e}")
        print("Saving as GIF instead...")
        imageio.mimsave(out_video_gif, frames, fps=fps)
        print(f"Video saved successfully to: {out_video_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LLG Simulation Video")
    parser.add_argument("--phase", type=str, default="SkX", help="Phase to initialize (SkX, SC, SP)")
    parser.add_argument("--L", type=int, default=32, help="Lattice size")
    parser.add_argument("--H", type=float, default=1.5, help="Scaled Magnetic Field (H)")
    parser.add_argument("--A", type=float, default=1.0, help="Scaled Anisotropy (A_s)")
    parser.add_argument("--steps", type=int, default=None, help="Total maximum steps to simulate. If None, it scales dynamically like the real solver.")
    parser.add_argument("--chunk", type=int, default=25, help="Steps per video frame")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video")
    parser.add_argument("--mode", type=str, choices=["quiver", "fft"], default="quiver", help="Visualization mode (quiver or fft)")
    
    args = parser.parse_args()
    make_video(args.phase, args.L, args.H, args.A, args.steps, args.chunk, args.fps, args.mode)
