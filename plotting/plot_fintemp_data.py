import os
import sys
# Allow imports from the parent directory (project root) when running this script directly or as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import re
import numpy as np
from phase_diagram import plot_phase_diagram, plot_energy_difference, plot_fm_stabilization_energy

def main():
    # Setup directories
    data_dir = "output/phase_diagrams/fintemp"
    existing_files = glob.glob(f"{data_dir}/*.npz")

    if not existing_files:
        print(f"Error: No finite-temperature phase diagram data found in '{data_dir}'.")
        if os.path.basename(os.getcwd()) == "plotting":
            print("\n[TIP] It looks like you ran this script from inside the 'plotting' folder.")
            print("Please navigate to the project root directory first, then run:")
            print("  cd ..")
            print("  python .\\plotting\\plot_fintemp_data.py")
        else:
            print("Please ensure your downloaded .npz file is placed in that folder.")
        sys.exit(1)

    print("\n=== Found Finite-Temperature Phase Diagram Data ===")
    for idx, f in enumerate(existing_files, 1):
        print(f" [{idx}] : {os.path.basename(f)}")

    # Prompt user for choice
    try:
        choice = input(f"\nWhich one would you like to plot? [1-{len(existing_files)}] (default 1): ").strip()
        if not choice:
            choice = "1"
        
        if choice.isdigit() and 0 < int(choice) <= len(existing_files):
            sel_file = existing_files[int(choice)-1]
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
    except (KeyboardInterrupt, SystemExit):
        print("\nExiting.")
        sys.exit(0)

    print(f"\nLoading data from: {os.path.basename(sel_file)}")
    data = np.load(sel_file)

    grid = data['grid']
    H_vals = data['H_vals']
    A_vals = data['A_vals']
    
    # Safely retrieve parameters
    T_sel = float(data['T']) if 'T' in data else 0.1
    
    # Try to parse L from file contents or filename
    L = 32
    if 'L' in data:
        L = int(data['L'])
    else:
        match = re.search(r'_L(\d+)_', os.path.basename(sel_file))
        if match:
            L = int(match.group(1))

    total_pts = len(H_vals) * len(A_vals)

    # Ensure output graph directory exists
    os.makedirs("output/phase_diagrams/fintemp", exist_ok=True)
    
    # Redirect directory path: plot_phase_diagram naturally saves into 'output/phase_diagrams/llg/{out_name}'.
    # We pass '../../phase_diagrams/fintemp' as a relative prefix so it seamlessly writes to 'output/phase_diagrams/fintemp/'.
    redirect_dir = "../../phase_diagrams/fintemp"

    pd_title = f"Topological Magnetic Phase Diagram (T = {T_sel})"
    out_pd_name = f"{redirect_dir}/fintemp_pd_T{T_sel}_L{L}_{total_pts}.png"
    
    print("\nGenerating Phase Diagram Plot...")
    plot_phase_diagram(grid, H_vals, A_vals, out_name=out_pd_name, title=pd_title)

    if 'energy_SkX' in data:
        energies_dict = {
            'SkX': data['energy_SkX'],
            'SC': data['energy_SC'],
            'SP': data['energy_SP'],
            'FM': data['energy_FM']
        }
        
        print("Generating Energy Difference Plot...")
        out_diff_name = f"{redirect_dir}/fintemp_energy_diff_T{T_sel}_L{L}_{total_pts}.png"
        ed_title = f"Energy Gap to First Excited Phase (T = {T_sel})"
        plot_energy_difference(energies_dict, H_vals, A_vals, out_name=out_diff_name, title=ed_title)
        
        print("Generating FM Stabilization Energy Plot...")
        out_fm_name = f"{redirect_dir}/fintemp_fm_stabilization_T{T_sel}_L{L}_{total_pts}.png"
        fm_title = f"FM Stabilization Energy (T = {T_sel})"
        plot_fm_stabilization_energy(energies_dict, H_vals, A_vals, out_name=out_fm_name, title=fm_title)

    print(f"\nSuccess! All high-res plots saved in: 'output/phase_diagrams/fintemp/'")

if __name__ == "__main__":
    main()
