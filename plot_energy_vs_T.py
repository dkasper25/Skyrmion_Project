import numpy as np
import matplotlib.pyplot as plt
import os
from fintemp_LLG import compare_fintemp_phases

# Create a clean mock arguments class to pass to the function
class Args:
    def __init__(self, T):
        self.L = 32           # Base unit cell size
        self.L_super = 64   # Supercell size for thermodynamics
        self.H = 1
        self.A = 0.5
        self.T = T
        self.dt = 0.005
        self.steps = 2000     # Reduced steps for a faster sweep
        self.block = 50
        self.seed = 42
        self.no_plot = True   # Disable live plotting so it sweeps instantly
        self.live_mode = "quiver"

def plot_energy_scaling():
    print("Starting Setup: Energy vs Temperature Sweep...")
    # Temperatures from almost 0 to 0.8
    temperatures = np.linspace(0.01, 20, 10) 
    
    phase_names = ["SkX", "SC", "SP", "FM"]
    energies_all = {p: [] for p in phase_names}
    energies_terms = {p: {'ex': [], 'dmi': [], 'z': [], 'a': []} for p in phase_names}

    import contextlib
    for T in temperatures:
        print(f"Sweeping at T = {T:.3f}... ", end="", flush=True)
        
        args = Args(T)
        # We set save_outputs=False so it doesn't flood your drive with .npz files during the sweep
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            winner, results, results_terms = compare_fintemp_phases(args, save_outputs=False)
        
        print("Energies: ", end="")
        for p in phase_names:
            val = results.get(p, np.nan)
            energies_all[p].append(val)
            
            terms = results_terms.get(p, None)
            if terms is not None:
                energies_terms[p]['ex'].append(terms['ex'])
                energies_terms[p]['dmi'].append(terms['dmi'])
                energies_terms[p]['z'].append(terms['z'])
                energies_terms[p]['a'].append(terms['a'])
            else:
                energies_terms[p]['ex'].append(np.nan)
                energies_terms[p]['dmi'].append(np.nan)
                energies_terms[p]['z'].append(np.nan)
                energies_terms[p]['a'].append(np.nan)

            if not np.isnan(val):
                print(f"{p}: {val:.4f} | ", end="")
        print(f"Winner: {winner}")

    # Plot the results
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    colors = {'SkX': 'red', 'SC': 'blue', 'SP': 'green', 'FM': 'purple'}
    markers = {'SkX': 'o', 'SC': 's', 'SP': '^', 'FM': 'D'}
    
    plot_keys = ['total', 'ex', 'dmi', 'a', 'z']
    titles = ['Total Thermal Energy Density', 'Normalized Exchange Geometry', 'Normalized DMI Geometry', 'Anisotropy Energy', 'Zeeman Energy']
    
    for idx, (key, title) in enumerate(zip(plot_keys, titles)):
        ax = axs[idx]
        for p in phase_names:
            if key == 'total':
                y_data = energies_all[p]
            else:
                y_data = energies_terms[p][key]
            
            ax.plot(temperatures, y_data, marker=markers[p], color=colors[p], 
                     label=p, markersize=8, linewidth=2, alpha=0.8, fillstyle='none', markeredgewidth=2)
            
        ax.set_xlabel("Scaled Temperature (T)", fontsize=12)
        if key == 'total':
            ax.set_ylabel("Energy Density", fontsize=12)
        else:
            ax.set_ylabel("Normalized Bond Value", fontsize=12)
            
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.7)
        if idx == 0:
            ax.legend(title="Magnetic Phase")
            
    # Hide the 6th empty subplot
    axs[5].axis('off')

    plt.suptitle(f"Phase Energy Comparison (H={args.H}, A={args.A})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and show
    os.makedirs("output/Fintemp", exist_ok=True)
    outpath = f"output/Fintemp/energy_vs_temperature_H{args.H}_A{args.A}_Tmax{args.T}.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nSweep Complete! Plot saved to '{outpath}'")
    
    plt.show()

if __name__ == "__main__":
    plot_energy_scaling()
