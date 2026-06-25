import os
import sys
# Allow imports from the parent directory (project root) when running this script directly or as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from LLG_solver import init_SkX, init_SC, init_SP, relax_phase_numba

def step_by_step_classification(spins, ax_val, ay_val, outdir="output/plots/llg/classification_steps", show_labels=True):
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving step-by-step visualizations to: {outdir}")
    
    # Step 1: Real space (Raw)
    n_z_raw = spins[:, :, 2]
    
    fig, ax = plt.subplots(figsize=(6,6))
    L_x, L_y = n_z_raw.shape
    extent = [-0.5 * L_x * ax_val, 0.5 * L_x * ax_val, -0.5 * L_y * ay_val, 0.5 * L_y * ay_val]
    ax.imshow(n_z_raw.T, cmap='bwr', extent=extent, origin='lower', vmin=-1.0, vmax=1.0)
    if show_labels:
        ax.set_title("Step 1: Real Space (Raw $n_z$)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/Step1_RealSpace.png", dpi=150)
    plt.close()
    
    # Step 2: Smoothing/Blur applied
    n_smoothed = scipy.ndimage.gaussian_filter(spins, sigma=[1.0, 1.0, 0], mode='wrap')
    norm = np.linalg.norm(n_smoothed, axis=-1, keepdims=True)
    n = n_smoothed / np.where(norm > 1e-12, norm, 1.0)
    n_z = n[:, :, 2]
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(n_z.T, cmap='bwr', extent=extent, origin='lower', vmin=-1.0, vmax=1.0)
    if show_labels:
        ax.set_title("Step 2: Real Space (Gaussian Blur Applied)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/Step2_RealSpace_Blurred.png", dpi=150)
    plt.close()
    
    # Step 3: Raw FFT Power Spectrum
    Mz = np.mean(n_z)
    fft_z = np.fft.fftshift(np.fft.fft2(n_z - Mz))
    power = np.abs(fft_z)**2
    power[L_x//2, L_y//2] = 0.0 # Remove DC
    
    I_x, I_y = np.indices(power.shape)
    k_x_real = (I_x - (L_x // 2)) * (2 * np.pi / (L_x * ax_val))
    k_y_real = (I_y - (L_y // 2)) * (2 * np.pi / (L_y * ay_val))
    
    dk_x = 2 * np.pi / (L_x * ax_val)
    dk_y = 2 * np.pi / (L_y * ay_val)
    extent_k = [
        (-L_x//2 - 0.5) * dk_x,
        (L_x - 1 - L_x//2 + 0.5) * dk_x,
        (-L_y//2 - 0.5) * dk_y,
        (L_y - 1 - L_y//2 + 0.5) * dk_y
    ]
    
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(np.log10(power.T + 1e-12), cmap='magma', extent=extent_k, origin='lower', interpolation='nearest')
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    if show_labels:
        ax.set_title("Step 3: FFT Power Spectrum (DC removed)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Log(Power)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/Step3_FFT_Raw.png", dpi=150)
    plt.close()
    
    # Step 4: Max Peak Detection (Thresholding)
    max_power = np.max(power)
    mask = power > (max_power * 0.25)
    
    # Treat each individual pixel above threshold as a peak (no connected components)
    peaks_indices = np.argwhere(mask)
    num_peaks = len(peaks_indices)
    
    r_k = np.sqrt(k_x_real**2 + k_y_real**2)
    power_no_dc = power.copy()
    power_no_dc[r_k < 1e-5] = 0.0
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(np.log10(power.T + 1e-12), cmap='magma', extent=extent_k, origin='lower', interpolation='nearest')
    
    if num_peaks > 0:
        k_x_peaks = (peaks_indices[:, 0] - (L_x // 2)) * dk_x
        k_y_peaks = (peaks_indices[:, 1] - (L_y // 2)) * dk_y
        
        max_idx = np.unravel_index(np.argmax(power_no_dc), power.shape)
        k_x_max = k_x_real[max_idx]
        k_y_max = k_y_real[max_idx]
        
        # Plot other peaks
        ax.scatter(k_x_peaks, k_y_peaks, s=60, color='lime', alpha=0.9, marker='o', edgecolors='black', linewidths=1, label="Detected Peaks (>25% Max)")
        # Plot highest peak
        ax.scatter(k_x_max, k_y_max, s=90, color='cyan', alpha=0.9, marker='o', edgecolors='black', linewidths=1, label="Highest Peak")
        if show_labels:
            ax.legend(loc="upper right", fontsize=8)
        
        r_fundamental_step4 = np.sqrt(k_x_max**2 + k_y_max**2)
        ax.set_xlim(-r_fundamental_step4 * 5.5, r_fundamental_step4 * 5.5)
        ax.set_ylim(-r_fundamental_step4 * 5.5, r_fundamental_step4 * 5.5)

    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    if show_labels:
        ax.set_title(f"Step 4: Peak Detection\n(Threshold > 25% Max, Found {num_peaks} peaks)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/Step4_Peak_Detection.png", dpi=150)
    plt.close()
    
    # Step 5: Fundamental Ring Isolation
    fig, ax = plt.subplots(figsize=(6,6))
    
    if np.max(power_no_dc) > 0:
        max_idx = np.unravel_index(np.argmax(power_no_dc), power.shape)
        r_fundamental = r_k[max_idx]
        ring_mask = (r_k > r_fundamental * 0.8) & (r_k < r_fundamental * 1.2)
        peak_mask = power_no_dc > (np.max(power_no_dc) * 0.25)
        active_mask = ring_mask & peak_mask
        
        # Visualizing the Bandpass: keep only the active bins (ring + peak threshold), dark background
        power_ring = np.zeros_like(power)
        power_ring[active_mask] = power[active_mask]
        
        # Match colors to Step 4 so the intensity is completely identical
        v_min = np.min(np.log10(power.T + 1e-12))
        v_max = np.max(np.log10(power.T + 1e-12))
        
        ax.imshow(np.log10(power_ring.T + 1e-12), cmap='magma', extent=extent_k, origin='lower', interpolation='nearest', vmin=v_min, vmax=v_max)
        
        # Plot ring boundaries
        circle_inner = plt.Circle((0,0), r_fundamental*0.8, color='white', fill=False, linestyle='--', alpha=0.5)
        circle_outer = plt.Circle((0,0), r_fundamental*1.2, color='white', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(circle_inner)
        ax.add_patch(circle_outer)
        
        # Highlight thresholded bins within the ring
        k_x_active = k_x_real[active_mask]
        k_y_active = k_y_real[active_mask]
        
        ax.set_xlim(-r_fundamental * 5.5, r_fundamental * 5.5)
        ax.set_ylim(-r_fundamental * 5.5, r_fundamental * 5.5)
        if show_labels:
            ax.set_title(f"Step 5: Radial Bandpass Filter\n(r = {r_fundamental:.3f} ± 20%)")
    else:
        active_mask = r_k > 1e-5
        ax.imshow(np.log10(power.T + 1e-12), cmap='magma', extent=extent_k, origin='lower', interpolation='nearest')
        if show_labels:
            ax.set_title("Step 5: Radial Bandpass Filter (Uniform Phase)")
        
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    plt.tight_layout()
    plt.savefig(f"{outdir}/Step5_Ring_Isolation.png", dpi=150)
    plt.close()
    
    # Step 6: Angular Power Spectrum Extraction
    angles = np.arctan2(k_y_real[active_mask], k_x_real[active_mask])
    weights = power[active_mask]
    
    c0 = np.sum(weights)
    if c0 > 0:
        c2 = np.abs(np.sum(weights * np.exp(-1j * 2 * angles))) / c0
        c4 = np.abs(np.sum(weights * np.exp(-1j * 4 * angles))) / c0
        c6 = np.abs(np.sum(weights * np.exp(-1j * 6 * angles))) / c0
    else:
        c2, c4, c6 = 0, 0, 0
        
    angles_deg = np.degrees(np.mod(angles, 2 * np.pi))
    hist, bin_edges = np.histogram(angles_deg, bins=180, range=(0, 360), weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(bin_centers, hist, color='purple', linewidth=2)
    ax.fill_between(bin_centers, hist, color='purple', alpha=0.3)
    
    if show_labels:
        textstr = f"$C_2$ (Spiral): {c2:.3f}\n$C_4$ (Square): {c4:.3f}\n$C_6$ (Hex): {c6:.3f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
                
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Angle (Degrees)")
    ax.set_ylabel("Integrated Power")
    if show_labels:
        ax.set_title("Step 6: Angular Power Spectrum & Symmetries")
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/Step6_Angular_Spectrum.png", dpi=150)
    plt.close()

    print(f"Workflow visualization complete! Output written to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Step-by-Step Phase Classification Workflow")
    parser.add_argument("--phase", type=str, default="SkX", help="Phase to initialize (SkX, SC, SP)")
    parser.add_argument("--L", type=int, default=32, help="Lattice size")
    parser.add_argument("--H", type=float, default=1.5, help="Scaled Magnetic Field (H)")
    parser.add_argument("--A", type=float, default=1.0, help="Scaled Anisotropy (A_s)")
    parser.add_argument("--no-labels", action="store_true", help="Turn off all plot titles, legends, axis labels, and text overlays")
    
    args = parser.parse_args()
    
    print(f"Relaxing {args.phase} for a final configuration to analyze...")
    if args.phase == "SkX":
        spins, ax_val, ay_val = init_SkX(args.L)
    elif args.phase == "SC":
        spins, ax_val, ay_val = init_SC(args.L)
    elif args.phase == "SP":
        spins, ax_val, ay_val = init_SP(args.L)
    else:
        raise ValueError("Invalid phase")
        
    spins, f_tot, ax_val, ay_val, _, _, _ = relax_phase_numba(
        spins, args.L, args.H, args.A, 5000, 1e-8, ax_val, ay_val, 0.0, 0.05, 0.25, global_step_start=0, iso_scale=False
    )
    
    outd = f"output/plots/llg/classification_steps/Classification_Steps_{args.phase}"
    step_by_step_classification(spins, ax_val, ay_val, outdir=outd, show_labels=not args.no_labels)
