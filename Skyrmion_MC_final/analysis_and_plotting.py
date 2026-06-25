
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter  # Added for smooth Gaussian filtering
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR  = "phase_diagram_configurations"
OUTPUT_DIR = "phase_diagram_plots"

# Updated standard python colors per your request:
# 0: FM (Light Grey), 1: Spiral (Orange), 2: Hex SkX (Blue), 3: Sq SkX (Green)
PHASE_COLORS = ['#f0f0f0', 'orange', 'blue', 'green']
PHASE_LABELS = ['FM', 'Spiral', 'Hex SkX', 'Sq SkX']
# ==========================================

def calculate_q(spins):
    """Calculates the topological charge Q for a 2D spin lattice."""
    L = spins.shape[0]
    total_q = 0.0
    for i in range(L):
        for j in range(L):
            ip, jp = (i + 1) % L, (j + 1) % L
            s00, s10, s01, s11 = spins[i, j], spins[ip, j], spins[i, jp], spins[ip, jp]
            
            # Triangle 1
            num1 = np.dot(s00, np.cross(s10, s01))
            den1 = 1 + np.dot(s00, s10) + np.dot(s10, s01) + np.dot(s01, s00)
            total_q += 2 * np.arctan2(num1, den1)
            
            # Triangle 2
            num2 = np.dot(s11, np.cross(s01, s10))
            den2 = 1 + np.dot(s11, s01) + np.dot(s01, s10) + np.dot(s10, s11)
            total_q += 2 * np.arctan2(num2, den2)
            
    return total_q / (4.0 * np.pi)

def analyze_phase(spins, h_val):
    """Classifies the magnetic phase using Sz and Fourier Analysis."""
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

def process_file(file_path):
    print(f"Processing: {os.path.basename(file_path)}")
    data = np.load(file_path, allow_pickle=True)
    
    spins_grid = data['spins']
    h_range = data['h_range']
    a_range = data['a_range']
    temp = data['temperature']
    res_h, res_a = spins_grid.shape[0], spins_grid.shape[1]
    
    q_map = np.zeros((res_h, res_a))
    phase_map = np.zeros((res_h, res_a))
    
    for i in range(res_h):
        for j in range(res_a):
            s = spins_grid[i, j]
            q_map[i, j] = calculate_q(s)
            phase_map[i, j] = analyze_phase(s, h_range[i])
            
    base_name = os.path.basename(file_path).replace("_data.npz", "")
    extent = [a_range[0], a_range[-1], h_range[-1], h_range[0]]
    
    # 1. AUTO-PHASE PLOT (AESTHETIC UPDATES)
    plt.figure(figsize=(7, 6))
    custom_cmap = ListedColormap(PHASE_COLORS)
    im = plt.imshow(phase_map, extent=extent, cmap=custom_cmap, aspect='auto', interpolation='nearest')
    
    # Configure Legend/Colorbar
    cbar = plt.colorbar(im, ticks=[0.375, 1.125, 1.875, 2.625]) # Centers labels on colors
    cbar.ax.set_yticklabels(PHASE_LABELS)
    
    # Updated Titles and Labels with LaTeX
    plt.title(fr"Topological Magnetic Phase Diagram ($T$ = {temp:.2f}, MC)", fontsize=12)
    plt.xlabel(r"Scaled Anisotropy ($A_S$)", fontsize=11)
    plt.ylabel(r"Scaled Magnetic Field ($H$)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_Phase_diagram.png"), dpi=200)
    plt.close()

    # 2. TOPOLOGICAL CHARGE PLOT (GAUSSIAN FILTER & VIRIDIS INTERPOLATION)
    plt.figure(figsize=(7, 6))
    
    # Smooth the raw topological charge map data with a Gaussian filter (sigma=1.0)
    smoothed_q_map = gaussian_filter(np.abs(q_map), sigma=1.0)
    
    # Using 'viridis' colormap and 'bicubic' interpolation for visual smoothness between bins
    plt.imshow(smoothed_q_map, extent=extent, cmap='viridis', aspect='auto', interpolation='bicubic')
    plt.colorbar(label=r"Topological Charge $|Q|$")
    plt.title(fr"Topological Charge Map ($T$ = {temp:.2f}, MC)")
    plt.xlabel(r"Scaled Anisotropy ($A_S$)")
    plt.ylabel(r"Scaled Magnetic Field ($H$)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_Q_map.png"), dpi=150)
    plt.close()

    # 3. VECTOR SUMMARY (BLUE-WHITE-RED TRANSITION)
    plt.figure(figsize=(14, 12))
    L = data['L']
    big_img = np.zeros((res_h * L, res_a * L))
    for i in range(res_h):
        for j in range(res_a):
            big_img[i*L:(i+1)*L, j*L:(j+1)*L] = spins_grid[i, j, :, :, 2]
            
    # Using 'bwr' colormap bounded explicitly between -1.0 and 1.0.
    # +1.0 (aligned spins) maps to Red, -1.0 (opposing spins) maps to Blue, transitioning smoothly via White at 0.0.
    plt.imshow(big_img, cmap='bwr', extent=extent, aspect='auto', vmin=-1.0, vmax=1.0)
    plt.title(fr"Spin Configurations $S_z$ ($T$ = {temp:.2f}, MC)")
    plt.xlabel(r"Scaled Anisotropy ($A_S$)")
    plt.ylabel(r"Scaled Magnetic Field ($H$)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_Vector_summary.png"), dpi=300)
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    files = glob.glob(os.path.join(INPUT_DIR, "*.npz"))
    if not files:
        print(f"No files found in {INPUT_DIR}")
        return
    
    for f in files:
        process_file(f)
    print("Aesthetic processing complete.")

if __name__ == "__main__":
    main()