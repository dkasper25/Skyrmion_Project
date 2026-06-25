import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm

def analyze_configuration(L_size, h_coeff, a_coeff):
    # 1. Path Setup
    config_dir = "configurations"
    real_space_dir = "plots_real_space"
    sf_dir = "plots_structure_factor"
    
    # Create plot directories if they don't exist
    for folder in [real_space_dir, sf_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    data_file = f"data_{L_size}_{h_coeff}_{a_coeff}.npz"
    data_path = os.path.join(config_dir, data_file)

    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return

    # 2. Load Data
    print(f"Loading {data_file}...")
    data = np.load(data_path)
    sz = data['relaxed_state'][:, :, 2]

    # --- PLOT 1: REAL SPACE CONFIGURATION ---
    print("Generating Real Space plot...")
    plt.figure(figsize=(8, 7))
    im1 = plt.imshow(sz.T, cmap='bwr', vmin=-1, vmax=1, origin='lower')
    
    # Using raw strings (r"...") to avoid LaTeX/Python backslash conflicts
    plt.title(r"Real Space Lattice ($S_z$)" + f"\nL={L_size}, h={h_coeff}, a={a_coeff}", fontsize=14)
    plt.colorbar(im1, label=r"Spin Component $S_z$", fraction=0.046, pad=0.04)
    plt.axis('off')
    
    real_space_path = os.path.join(real_space_dir, f"real_space_{L_size}_{h_coeff}_{a_coeff}.png")
    plt.savefig(real_space_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {real_space_path}")

    # --- PLOT 2: STRUCTURE FACTOR ---
    print("Calculating and plotting Structure Factor...")
    sz_dm = sz - np.mean(sz)
    
    ft = np.fft.fft2(sz_dm)
    ft_shift = np.fft.fftshift(ft)
    s_k = np.abs(ft_shift)**2

    plt.figure(figsize=(8, 7))
    
    # Define extent to stretch axes from -pi to pi
    k_extent = [-np.pi, np.pi, -np.pi, np.pi]
    
    # LogNorm is essential for seeing the 6-fold symmetry clearly
    im2 = plt.imshow(s_k, cmap='viridis', origin='lower', 
                     norm=LogNorm(vmin=1e-1), extent=k_extent)
    
    plt.title(r"Structure Factor $S(\mathbf{k})$" + f"\nL={L_size}, h={h_coeff}, a={a_coeff}", fontsize=14)
    plt.colorbar(im2, label=r"Intensity $|S(\mathbf{k})|^2$", fraction=0.046, pad=0.04)
    
    # Explicitly label the reciprocal space axes
    plt.xlabel(r"$k_x$", fontsize=12)
    plt.ylabel(r"$k_y$", fontsize=12)
    
    # Setting uniform tick distributions from -pi to pi
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    
    plt.grid(False) # Ensures default grid lines don't overlay the image unexpectedly
    
    sf_path = os.path.join(sf_dir, f"structure_factor_{L_size}_{h_coeff}_{a_coeff}.png")
    plt.savefig(sf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sf_path}")

if __name__ == "__main__":
    # --- ADJUST THESE PARAMETERS TO MATCH YOUR FILE ---
    L = 31
    H = 0.75
    A = 0.75
    # --------------------------------------------------
    
    analyze_configuration(L, H, A)