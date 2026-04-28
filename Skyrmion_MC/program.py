import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from simulation import run_simulation, run_heating_step
import multiprocessing as mp
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
L_SPINS            = 31      
RES                = 20      

# Cooling Protocol
T_START_COOL       = 1.0     
T_MIN              = 0.01    
COOL_STEPS         = 10000   

# Heating Protocol
HEATING_CYCLES     = 1       
HEAT_STEPS_PER_CYC = 10000    
TEMP_INCREMENT     = 0.5     

H_RANGE = (0.0, 2.5)         
A_RANGE = (-1.5, 1.7)        
# ==========================================

# --- PHYSICS & CLASSIFICATION HELPERS ---

def calculate_q(spins):
    L = spins.shape[0]
    total_q = 0.0
    for i in range(L):
        for j in range(L):
            ip, jp = (i + 1) % L, (j + 1) % L
            s00, s10, s01, s11 = spins[i, j], spins[ip, j], spins[i, jp], spins[ip, jp]
            num1 = np.dot(s00, np.cross(s10, s01))
            den1 = 1 + np.dot(s00, s10) + np.dot(s10, s01) + np.dot(s01, s00)
            total_q += 2 * np.arctan2(num1, den1)
            num2 = np.dot(s11, np.cross(s01, s10))
            den2 = 1 + np.dot(s11, s01) + np.dot(s01, s10) + np.dot(s10, s11)
            total_q += 2 * np.arctan2(num2, den2)
    return total_q / (4.0 * np.pi)

def analyze_phase(spins, h_val):
    Sz = spins[:, :, 2]
    avg_sz = np.abs(np.mean(Sz))
    if avg_sz > 0.85: return 0 
    if h_val < 0.25 and avg_sz < 0.4: return 1

    Sz_centered = Sz - np.mean(Sz)
    Sq = np.abs(np.fft.fftshift(np.fft.fft2(Sz_centered)))**2
    h_idx, w_idx = Sq.shape
    cy, cx = h_idx // 2, w_idx // 2
    y, x = np.indices((h_idx, w_idx))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta = np.arctan2(y - cy, x - cx)

    radial_sum = np.bincount(r.astype(int).ravel(), weights=Sq.ravel())
    if len(radial_sum) < 10: return 2
    q_star = np.argmax(radial_sum[5:]) + 5 

    mask = (r >= q_star - 2) & (r <= q_star + 2)
    counts, _ = np.histogram(theta[mask], bins=np.linspace(-np.pi, np.pi, 73), weights=Sq[mask])
    ang_fft = np.abs(np.fft.fft(counts - np.mean(counts)))
    return 3 if ang_fft[4] > ang_fft[6] else 2

# --- PLOTTING HELPERS ---

def format_time(seconds):
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def save_all_plots(spins_list, q_list, phase_list, stage_label, temp):
    base_name = f"L{L_SPINS}_Res{RES}_T{temp:.2f}_{stage_label}"
    plot_extent = [A_RANGE[0], A_RANGE[1], H_RANGE[0], H_RANGE[1]]

    # 1. Vector Map
    fig_dim = max(12, RES * 1.5)
    fig, axes = plt.subplots(RES, RES, figsize=(fig_dim, fig_dim))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    if RES == 1: axes = np.array([[axes]])
    for idx, s in enumerate(spins_list):
        i, j = divmod(idx, RES)
        axes[i, j].imshow(s[:, :, 2].T, cmap='bwr', vmin=-1, vmax=1, origin='lower')
        skip = max(1, L_SPINS // 10)
        axes[i, j].quiver(np.arange(0, L_SPINS, skip), np.arange(0, L_SPINS, skip), 
                          s[::skip, ::skip, 0].T, s[::skip, ::skip, 1].T, color='black', pivot='mid', scale=L_SPINS*1.2)
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    plt.savefig(os.path.join("outputs", f"{base_name}_vector.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)

    # 2. Q Heatmap
    q_matrix = np.array(q_list).reshape((RES, RES))
    plt.figure(figsize=(8, 6))
    im_q = plt.imshow(q_matrix, extent=plot_extent, origin='upper', cmap='magma', interpolation='gaussian', aspect='auto')
    plt.colorbar(im_q, label='Q')
    plt.title(f"Q Heatmap: {stage_label} (T={temp:.2f})")
    plt.savefig(os.path.join("outputs", f"{base_name}_qmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Auto-Phase Diagram
    phase_matrix = np.array(phase_list).reshape((RES, RES))
    cmap = ListedColormap(['#ff4d4d', '#4d79ff', '#2eb82e', '#ffcc00'])
    labels = ['FM', 'Stripe', 'Hex SkX', 'Sq SkX']
    plt.figure(figsize=(8, 6))
    im_p = plt.imshow(phase_matrix, extent=plot_extent, origin='upper', cmap=cmap, aspect='auto')
    cbar = plt.colorbar(im_p, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(labels)
    plt.title(f"Auto-Phase: {stage_label} (T={temp:.2f})")
    plt.savefig(os.path.join("outputs", f"{base_name}_autophase.png"), dpi=150, bbox_inches='tight')
    plt.close()

# --- WORKERS ---

def cooling_worker(args):
    h, a = args
    spins = run_simulation(L_SPINS, 1.0, 0.5, h, a, T_START_COOL, T_MIN, COOL_STEPS)
    return spins, calculate_q(spins), analyze_phase(spins, h)

def heating_worker(args):
    spins, t_start, t_end, h, a = args
    new_spins = run_heating_step(spins, t_start, t_end, L_SPINS, 1.0, 0.5, h, a, HEAT_STEPS_PER_CYC)
    return new_spins, calculate_q(new_spins), analyze_phase(new_spins, h)

# --- MAIN EXECUTION ---

def main():
    if not os.path.exists("outputs"): os.makedirs("outputs")
    h_bins = np.linspace(H_RANGE[1], H_RANGE[0], RES) 
    a_bins = np.linspace(A_RANGE[0], A_RANGE[1], RES)
    tasks_base = [(h, a) for h in h_bins for a in a_bins]
    
    num_cores = mp.cpu_count()
    start_time = time.time()
    total_phases = 1 + HEATING_CYCLES # 1 (Cooling) + Heating Cycles

    # INITIAL HEADER
    print(f"--- SIMULATION START ---")
    print(f"Running on: {num_cores} cores")
    print(f"Start Time: {time.strftime('%H:%M')}")
    print(f"Total Phases to simulate: {total_phases}")
    print(f"-------------------------")

    current_spins = [None] * len(tasks_base)
    current_qs = [None] * len(tasks_base)
    current_phases = [None] * len(tasks_base)

    # 1. COOLING
    print(f"\n[PHASE 1/{total_phases}: COOLING] T={T_MIN}")
    with mp.Pool(processes=num_cores) as pool:
        for idx, (s, q, p) in enumerate(pool.imap(cooling_worker, tasks_base), 1):
            current_spins[idx-1], current_qs[idx-1], current_phases[idx-1] = s, q, p
            if idx % 10 == 0 or idx == len(tasks_base):
                perc = (idx / len(tasks_base)) * 100
                print(f"  > Progress: {idx}/{len(tasks_base)} ({perc:.1f}%) | Time: {format_time(time.time()-start_time)}")
    save_all_plots(current_spins, current_qs, current_phases, "0_cooled", T_MIN)

    # 2. HEATING
    curr_t = T_MIN
    for cycle in range(1, HEATING_CYCLES + 1):
        next_t = curr_t + TEMP_INCREMENT
        phase_num = cycle + 1
        print(f"\n[PHASE {phase_num}/{total_phases}: HEATING CYCLE {cycle}] T={next_t:.2f}")
        
        heat_tasks = [(current_spins[i], curr_t, next_t, tasks_base[i][0], tasks_base[i][1]) for i in range(len(tasks_base))]
        
        with mp.Pool(processes=num_cores) as pool:
            for idx, (s, q, p) in enumerate(pool.imap(heating_worker, heat_tasks), 1):
                current_spins[idx-1], current_qs[idx-1], current_phases[idx-1] = s, q, p
                if idx % 10 == 0 or idx == len(tasks_base):
                    perc = (idx / len(tasks_base)) * 100
                    print(f"  > Progress: {idx}/{len(tasks_base)} ({perc:.1f}%) | Time: {format_time(time.time()-start_time)}")
        
        save_all_plots(current_spins, current_qs, current_phases, f"cycle_{cycle}", next_t)
        curr_t = next_t

    # FINAL FOOTER
    print(f"\n-------------------------")
    print(f"All tasks were completed in {format_time(time.time()-start_time)}")
    print(f"Results saved in 'outputs/' folder.")

if __name__ == '__main__':
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()