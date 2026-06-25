import numpy as np
from simulation_phase_diagram import run_simulation, run_heating_step
import multiprocessing as mp
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
L_SPINS            = 15      
RES                = 20      

# Cooling Protocol
T_START_COOL       = 1.0     
T_MIN              = 0.01    
COOL_STEPS         = 10000   

# Heating Protocol
HEATING_CYCLES     = 0       
HEAT_STEPS_PER_CYC = 100000    
TEMP_INCREMENT     = 0.25     

H_RANGE = (0.0, 2.5)         
A_RANGE = (-1.5, 1.7)        

# Output Directory
OUTPUT_DIR         = "phase_diagram_configurations"
# ==========================================

# --- DATA SAVING HELPER ---

def save_raw_data(spins_list, stage_label, temp, h_bins, a_bins):
    """
    Saves the raw spin configurations and parameters to a compressed .npz file.
    """
    base_name = f"L{L_SPINS}_Res{RES}_T{temp:.2f}_{stage_label}"
    file_path = os.path.join(OUTPUT_DIR, f"{base_name}_data.npz")
    
    # Reshape the flat list into (H_res, A_res, L, L, 3)
    spins_grid = np.array(spins_list).reshape((RES, RES, L_SPINS, L_SPINS, 3))
    
    np.savez_compressed(
        file_path,
        spins=spins_grid,
        h_range=h_bins,
        a_range=a_bins,
        temperature=temp,
        L=L_SPINS,
        timestamp=time.time()
    )
    print(f"  [RAW DATA SAVED]: {file_path}")

# --- WORKERS ---

def cooling_worker(args):
    h, a = args
    spins = run_simulation(L_SPINS, 1.0, 0.5, h, a, T_START_COOL, T_MIN, COOL_STEPS)
    return spins

def heating_worker(args):
    spins, t_start, t_end, h, a = args
    new_spins = run_heating_step(spins, t_start, t_end, L_SPINS, 1.0, 0.5, h, a, HEAT_STEPS_PER_CYC)
    return new_spins

# --- MAIN EXECUTION ---

def main():
    # Create the custom folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    h_bins = np.linspace(H_RANGE[1], H_RANGE[0], RES) 
    a_bins = np.linspace(A_RANGE[0], A_RANGE[1], RES)
    tasks_base = [(h, a) for h in h_bins for a in a_bins]
    
    total_cores = mp.cpu_count()
    num_cores = max(1, total_cores - 2) # Leave 2 cores free for system responsiveness
    
    start_time = time.time()
    total_phases = 1 + HEATING_CYCLES

    print(f"--- RAW DATA SIMULATION START ---")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Using: {num_cores} cores | L={L_SPINS} | Resolution={RES}x{RES}")

    current_spins = [None] * len(tasks_base)

    # 1. COOLING PHASE
    print(f"\n[PHASE 1/{total_phases}: COOLING] T={T_MIN}")
    with mp.Pool(processes=num_cores) as pool:
        for idx, s in enumerate(pool.imap(cooling_worker, tasks_base), 1):
            current_spins[idx-1] = s
            if idx % 20 == 0 or idx == len(tasks_base):
                print(f"  > Progress: {idx}/{len(tasks_base)}")

    save_raw_data(current_spins, "cooled", T_MIN, h_bins, a_bins)

    # 2. HEATING PHASE
    curr_t = T_MIN
    for cycle in range(1, HEATING_CYCLES + 1):
        next_t = curr_t + TEMP_INCREMENT
        print(f"\n[PHASE {cycle+1}/{total_phases}: HEATING] T={next_t:.2f}")
        
        heat_tasks = [(current_spins[i], curr_t, next_t, tasks_base[i][0], tasks_base[i][1]) 
                      for i in range(len(tasks_base))]
        
        with mp.Pool(processes=num_cores) as pool:
            for idx, s in enumerate(pool.imap(heating_worker, heat_tasks), 1):
                current_spins[idx-1] = s
        
        save_raw_data(current_spins, f"cycle_{cycle}", next_t, h_bins, a_bins)
        curr_t = next_t

    print(f"\nSimulation complete. Total time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()