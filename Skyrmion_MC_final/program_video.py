import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time
import os
from simulation_video import run_video_protocol

# --- PARAMETERS ---
L_SIZE         = 31       
H_COEFF        = 0.75       
A_COEFF        = 0.75       
T_START, T_MIN, T_MAX = 1.0, 0.01, 2.0

STEPS_COOL     = 2000000     
STEPS_RELAX    = 1000000     
STEPS_HEAT     = 3000000    
FPS            = 20
VIDEO_LEN      = 25   
FRAME_EVERY    = int((STEPS_COOL+STEPS_RELAX+STEPS_HEAT)/(FPS*VIDEO_LEN))        

def main():
    # 1. Create directories for a clean environment
    config_dir = "configurations"
    video_dir = "videos"
    for folder in [config_dir, video_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    base_name = f"{L_SIZE}_{H_COEFF}_{A_COEFF}"
    
    # Define paths using os.path.join for cross-platform safety
    video_path = os.path.join(video_dir, f"protocol_{base_name}.mp4")
    data_path = os.path.join(config_dir, f"data_{base_name}.npz")
    
    print(f"--- Simulation Started: {base_name} ---")
    start_sim = time.time()
    
    frames_data = run_video_protocol(
        L_SIZE, H_COEFF, A_COEFF, 
        T_START, T_MIN, T_MAX, 
        STEPS_COOL, STEPS_RELAX, STEPS_HEAT, 
        FRAME_EVERY
    )
    
    all_energies = [f[2] for f in frames_data]
    all_temps = [f[1] for f in frames_data]
    num_frames = len(frames_data)
    
    cool_f  = STEPS_COOL // FRAME_EVERY
    relax_f = STEPS_RELAX // FRAME_EVERY
    
    # --- DATA SAVING (.npz) ---
    # We save everything first so the data is safe even if video encoding fails
    print(f"Saving spin configurations to {data_path}...")
    
    idx_relax_end = cool_f + relax_f - 1
    relaxed_spins = frames_data[idx_relax_end][0]
    
    heating_spins = np.array([f[0] for f in frames_data[idx_relax_end:]])
    heating_temps = np.array(all_temps[idx_relax_end:])
    
    np.savez_compressed(
        data_path, 
        relaxed_state=relaxed_spins, 
        heating_trajectory=heating_spins,
        temps=heating_temps,
        energies=np.array(all_energies)
    )

    # --- VIDEO ENCODING ---
    print(f"Encoding MP4 to {video_path}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
    
    # Initial frame
    im = ax1.imshow(frames_data[0][0][:, :, 2].T, cmap='bwr', vmin=-1, vmax=1, origin='lower')
    ax1.axis('off')
    
    line, = ax2.plot([], [], color='crimson', lw=2)
    ax2.set_xlim(0, num_frames)
    ax2.set_ylim(min(all_energies) - 0.02, max(all_energies) + 0.02)
    ax2.set_title("Energy Evolution")
    ax2.grid(True, alpha=0.3)
    
    phase_text = fig.text(0.5, 0.02, "", ha='center', fontsize=12, fontweight='bold')

    def update(i):
        spins, temp, energy = frames_data[i]
        im.set_array(spins[:, :, 2].T)
        ax1.set_title(f"T = {temp:.3f}")
        line.set_data(np.arange(i), all_energies[:i])
        
        if i < cool_f:
            txt = "PHASE 1: COOLING"
        elif i < (cool_f + relax_f):
            txt = "PHASE 2: RELAXATION"
        else:
            txt = "PHASE 3: HEATING / MELTING"
        phase_text.set_text(txt)
        return im, line, phase_text

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    writer = FFMpegWriter(fps=FPS, bitrate=2500)
    ani.save(video_path, writer=writer)
    plt.close()
    
    print(f"--- Finished! Simulation data and video are organized in their folders. ---")

if __name__ == "__main__":
    main()