import numpy as np
import scipy.ndimage
import os
import matplotlib.pyplot as plt

def analyze_state_llg(spins, ax, ay, phase_name="Unknown", plot_fft=False, outpath_override=None):
    """
    Computes Topological Charge (Q) and categorizes phase symmetry 
    by checking physical gradients and Fast Fourier Transform (FFT) Bragg peaks.
    
    This is the EXACT classification pipeline used in LLG_solver.py, 
    preserving identical deterministic parameters for LLG comparison compatibility.
    """
    # Light Gaussian blur to filter out high-frequency magnons
    n_smoothed = scipy.ndimage.gaussian_filter(spins, sigma=[1.0, 1.0, 0], mode='wrap')
    
    # Re-normalize after smoothing
    norm = np.linalg.norm(n_smoothed, axis=-1, keepdims=True)
    n = n_smoothed / np.where(norm > 1e-12, norm, 1.0)
    
    n_right = np.roll(n, -1, axis=0)
    n_left = np.roll(n, 1, axis=0)
    n_up = np.roll(n, -1, axis=1)
    n_down = np.roll(n, 1, axis=1)
    
    dn_dx = (n_right - n_left) / (2.0 * ax)
    dn_dy = (n_up - n_down) / (2.0 * ay)
    
    # 1. Topological Charge (Q)
    charge_density = np.sum(n * np.cross(dn_dx, dn_dy), axis=-1) * ax * ay
    Q = np.sum(charge_density) / (4.0 * np.pi)
    
    # 2. Geometry via FFT Analysis
    Mz = np.mean(n[:, :, 2])
    n_z = n[:, :, 2]
    
    fft_z = np.fft.fftshift(np.fft.fft2(n_z - Mz))
    power = np.abs(fft_z)**2
    
    L_x, L_y = n_z.shape
    power[L_x//2, L_y//2] = 0.0
    
    max_power = np.max(power)
    is_uniform = np.var(spins[:, :, 2]) < 1e-3
    
    num_peaks = 0
    mask = None
    geometry = "Uniform (FM)"
    
    c2, c4, c6 = 0.0, 0.0, 0.0
    
    if not is_uniform:
        # --- Angular Fourier Analysis for Symmetry ---
        I_x, I_y = np.indices(power.shape)
        
        # Convert raw indices to true physical k-space coordinates
        k_x_real = (I_x - (L_x // 2)) * (2 * np.pi / (L_x * ax))
        k_y_real = (I_y - (L_y // 2)) * (2 * np.pi / (L_y * ay))
        
        r_k = np.sqrt(k_x_real**2 + k_y_real**2)
        
        # 1. Isolate the fundamental Bragg peak radius
        power_no_dc = power.copy()
        power_no_dc[r_k < 1e-5] = 0.0
        
        if np.max(power_no_dc) > 0:
            max_idx = np.unravel_index(np.argmax(power_no_dc), power.shape)
            r_fundamental = r_k[max_idx]
            
            # 2. Analyze ONLY the fundamental k-space ring (Radial Bandpass)
            ring_mask = (r_k > r_fundamental * 0.8) & (r_k < r_fundamental * 1.2)
            
            # 3. Intensity Threshold
            peak_mask = power_no_dc > (np.max(power_no_dc) * 0.25)
            active_mask = ring_mask & peak_mask
        else:
            active_mask = r_k > 1e-5
            
        # Calculate true physical angles
        angles = np.arctan2(k_y_real[active_mask], k_x_real[active_mask])
        weights = power[active_mask]
        
        c0 = np.sum(weights)
        if c0 > 0:
            c2 = np.abs(np.sum(weights * np.exp(-1j * 2 * angles))) / c0
            c4 = np.abs(np.sum(weights * np.exp(-1j * 4 * angles))) / c0
            c6 = np.abs(np.sum(weights * np.exp(-1j * 6 * angles))) / c0
            
        mask = power > (max_power * 0.25)
        
        # Treat each individual pixel above threshold as a peak (no connected components)
        peaks_indices = np.argwhere(mask)
        num_peaks = len(peaks_indices)
        
        is_collinear = False
        if num_peaks > 0:
            dy = peaks_indices[:, 0] - (L_x // 2)
            dx = peaks_indices[:, 1] - (L_y // 2)
            angles_peaks = np.arctan2(dy, dx)
            R = np.sqrt(np.mean(np.cos(2 * angles_peaks))**2 + np.mean(np.sin(2 * angles_peaks))**2)
            if num_peaks == 1:
                is_collinear = True
            else:
                is_collinear = R > 0.9

        if is_collinear and (num_peaks % 2 == 0 or num_peaks == 1):
            geometry = "1D Spiral"
        elif c2 > c4 and c2 > c6 and c2 >= 0.15:
            geometry = "1D Spiral"
        elif c6 > c4 and c6 >= 0.15:
            geometry = "2D Hexagonal"
        elif c4 >= 0.15:
            geometry = "2D Square"
        else:
            geometry = "Unknown Geometry"
            
    # 3. Topology Classification
    topology = "Skyrmionic" if (not is_uniform and abs(Q) > 0.5) else "Trivial"
    
    # 4. Final Definitive State Mapping
    classified_state = "FM"
    if not is_uniform:
        # Morphological Shape Analysis to identify Labyrinthine Stripes
        mask = n[:, :, 2] > Mz
        labeled, num_features = scipy.ndimage.label(mask)
        shape_factor = 1.0
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            largest_label = np.argmax(sizes) + 1
            largest_mask = labeled == largest_label
            area = sizes[largest_label - 1]
            eroded = scipy.ndimage.binary_erosion(largest_mask)
            perimeter = np.sum(largest_mask & ~eroded)
            if perimeter > 0:
                shape_factor = (4 * np.pi * area) / (perimeter**2)
                
        if shape_factor < 0.15:
            classified_state = "SP"
            geometry = "Labyrinthine Stripe"
        else:
            if geometry == "2D Hexagonal" and topology == "Skyrmionic" and c6 >= 0.15:
                classified_state = "SkX"
            elif geometry == "2D Square" and c4 >= 0.15:
                classified_state = "SC"
            elif geometry == "1D Spiral" and abs(Q) < 0.5:
                classified_state = "SP"
            else:
                classified_state = "FM"
            
    if plot_fft:
        _plot_analysis_fft(n_z, power, is_uniform, num_peaks, peaks_indices, L_x, L_y, ax, ay, 
                           k_x_real, k_y_real, max_idx, r_fundamental, angles, weights, c2, c4, c6, 
                           geometry, phase_name, outpath_override)
        
    return {
        "Q": Q, 
        "Mz": Mz, 
        "geometry": geometry, 
        "topology": topology, 
        "peaks": num_peaks, 
        "is_uniform": is_uniform, 
        "classified_state": classified_state,
        "c2": c2 if not is_uniform else 0.0,
        "c4": c4 if not is_uniform else 0.0,
        "c6": c6 if not is_uniform else 0.0
    }

def analyze_state_mc(spins, ax, ay, phase_name="Unknown", plot_fft=False, outpath_override=None, sigma=1.5):
    """
    Computes Topological Charge (Q) and categorizes phase symmetry.
    
    OPTIMIZED FOR MONTE CARLO DATA:
    1. Increased Gaussian blur (sigma=1.5 by default) to filter out thermal high-frequency fluctuations.
    2. Robust hybrid twofold angular symmetry classification:
       A phase is classified as a 1D Spiral if c2 > 0.40 (strong twofold angular symmetry on the 
       fundamental reciprocal ring) OR if the discrete peaks are collinear.
    """
    # Enhanced spatial smoothing to reduce finite-temperature magnons
    n_smoothed = scipy.ndimage.gaussian_filter(spins, sigma=[sigma, sigma, 0], mode='wrap')
    
    # Re-normalize after smoothing
    norm = np.linalg.norm(n_smoothed, axis=-1, keepdims=True)
    n = n_smoothed / np.where(norm > 1e-12, norm, 1.0)
    
    n_right = np.roll(n, -1, axis=0)
    n_left = np.roll(n, 1, axis=0)
    n_up = np.roll(n, -1, axis=1)
    n_down = np.roll(n, 1, axis=1)
    
    dn_dx = (n_right - n_left) / (2.0 * ax)
    dn_dy = (n_up - n_down) / (2.0 * ay)
    
    # 1. Topological Charge (Q)
    charge_density = np.sum(n * np.cross(dn_dx, dn_dy), axis=-1) * ax * ay
    Q = np.sum(charge_density) / (4.0 * np.pi)
    
    # 2. Geometry via FFT Analysis
    Mz = np.mean(n[:, :, 2])
    n_z = n[:, :, 2]
    
    fft_z = np.fft.fftshift(np.fft.fft2(n_z - Mz))
    power = np.abs(fft_z)**2
    
    L_x, L_y = n_z.shape
    power[L_x//2, L_y//2] = 0.0
    
    max_power = np.max(power)
    is_uniform = np.var(spins[:, :, 2]) < 1e-3
    
    num_peaks = 0
    mask = None
    geometry = "Uniform (FM)"
    
    c2, c4, c6 = 0.0, 0.0, 0.0
    
    if not is_uniform:
        I_x, I_y = np.indices(power.shape)
        k_x_real = (I_x - (L_x // 2)) * (2 * np.pi / (L_x * ax))
        k_y_real = (I_y - (L_y // 2)) * (2 * np.pi / (L_y * ay))
        r_k = np.sqrt(k_x_real**2 + k_y_real**2)
        
        power_no_dc = power.copy()
        power_no_dc[r_k < 1e-5] = 0.0
        
        if np.max(power_no_dc) > 0:
            max_idx = np.unravel_index(np.argmax(power_no_dc), power.shape)
            r_fundamental = r_k[max_idx]
            ring_mask = (r_k > r_fundamental * 0.8) & (r_k < r_fundamental * 1.2)
            peak_mask = power_no_dc > (np.max(power_no_dc) * 0.25)
            active_mask = ring_mask & peak_mask
        else:
            active_mask = r_k > 1e-5
            
        angles = np.arctan2(k_y_real[active_mask], k_x_real[active_mask])
        weights = power[active_mask]
        
        c0 = np.sum(weights)
        if c0 > 0:
            c2 = np.abs(np.sum(weights * np.exp(-1j * 2 * angles))) / c0
            c4 = np.abs(np.sum(weights * np.exp(-1j * 4 * angles))) / c0
            c6 = np.abs(np.sum(weights * np.exp(-1j * 6 * angles))) / c0
            
        mask = power > (max_power * 0.25)
        peaks_indices = np.argwhere(mask)
        num_peaks = len(peaks_indices)
        
        is_collinear = False
        if num_peaks > 0:
            dy = peaks_indices[:, 0] - (L_x // 2)
            dx = peaks_indices[:, 1] - (L_y // 2)
            angles_peaks = np.arctan2(dy, dx)
            R = np.sqrt(np.mean(np.cos(2 * angles_peaks))**2 + np.mean(np.sin(2 * angles_peaks))**2)
            if num_peaks == 1:
                is_collinear = True
            else:
                is_collinear = R > 0.9
        
        # --- MC-OPTIMIZED GEOMETRY DECISION ---
        # A perfect stripe phase has c2 = c4 = c6 = 1.0. 
        # But Square (SC) has c2=0, c4=1, c6=0, and Hexagonal (SkX) has c2=0, c4=0, c6=1.
        # Thus, a large c2 (c2 > 0.40) is the mathematically perfect identifier of stripe symmetry.
        c2_stripe = (c2 > 0.40)
        
        if c2_stripe or (is_collinear and (num_peaks % 2 == 0 or num_peaks == 1)):
            geometry = "1D Spiral"
        elif c6 > c4:
            geometry = "2D Hexagonal"
        else:
            geometry = "2D Square"
            
    topology = "Skyrmionic" if (not is_uniform and abs(Q) > 0.5) else "Trivial"
    
    classified_state = "FM"
    if not is_uniform:
        # Morphological Shape Analysis to identify Labyrinthine Stripes
        mask = n_z > Mz
        labeled, num_features = scipy.ndimage.label(mask)
        shape_factor = 1.0
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            largest_label = np.argmax(sizes) + 1
            largest_mask = labeled == largest_label
            area = sizes[largest_label - 1]
            eroded = scipy.ndimage.binary_erosion(largest_mask)
            perimeter = np.sum(largest_mask & ~eroded)
            if perimeter > 0:
                shape_factor = (4 * np.pi * area) / (perimeter**2)
                
        if shape_factor < 0.15:
            classified_state = "SP"
            geometry = "Labyrinthine Stripe"
        else:
            if geometry == "2D Hexagonal":
                classified_state = "SkX"
            elif geometry == "2D Square":
                classified_state = "SC"
            elif geometry == "1D Spiral":
                classified_state = "SP"
            else:
                classified_state = "Unknown Phase"
            
    if plot_fft:
        _plot_analysis_fft(n_z, power, is_uniform, num_peaks, peaks_indices, L_x, L_y, ax, ay, 
                           k_x_real, k_y_real, max_idx, r_fundamental, angles, weights, c2, c4, c6, 
                           geometry, phase_name, outpath_override)
        
    return {
        "Q": Q, "Mz": Mz, "geometry": geometry, "topology": topology, 
        "peaks": num_peaks, "is_uniform": is_uniform, "classified_state": classified_state,
        "c2": c2, "c4": c4, "c6": c6
    }

def _plot_analysis_fft(n_z, power, is_uniform, num_peaks, peaks_indices, L_x, L_y, ax, ay, 
                       k_x_real, k_y_real, max_idx, r_fundamental, angles, weights, c2, c4, c6, 
                       geometry, phase_name, outpath_override):
    """Internal helper to render reciprocal power spectra."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    extent_real = [-0.5 * L_x * ax, 0.5 * L_x * ax, -0.5 * L_y * ay, 0.5 * L_y * ay]
    
    axs[0].imshow(n_z.T, cmap='bwr', extent=extent_real, origin='lower')
    axs[0].set_title(f"Real Space ($n_z$) - {phase_name}")
    axs[0].set_xlabel("X (Physical)")
    axs[0].set_ylabel("Y (Physical)")
    
    if is_uniform:
        axs[1].text(0.5, 0.5, "Uniform State\n(No Peaks)", ha='center', va='center')
        axs[1].axis('off')
        axs[2].text(0.5, 0.5, "Uniform State\n(No Symmetry)", ha='center', va='center')
        axs[2].axis('off')
    else:
        dk_x = 2 * np.pi / (L_x * ax)
        dk_y = 2 * np.pi / (L_y * ay)
        extent_k = [
            (-L_x//2 - 0.5) * dk_x,
            (L_x - 1 - L_x//2 + 0.5) * dk_x,
            (-L_y//2 - 0.5) * dk_y,
            (L_y - 1 - L_y//2 + 0.5) * dk_y
        ]
        
        im = axs[1].imshow(np.log10(power.T + 1e-12), cmap='magma', extent=extent_k, origin='lower', interpolation='nearest')
        axs[1].set_title(f"Reciprocal Power Spectrum (Log)\n{num_peaks} Fundamental Peaks")
        axs[1].set_xlabel(r"$k_x$")
        axs[1].set_ylabel(r"$k_y$")
        
        if num_peaks > 0:
            k_x_peaks = (peaks_indices[:, 0] - (L_x // 2)) * dk_x
            k_y_peaks = (peaks_indices[:, 1] - (L_y // 2)) * dk_y
            axs[1].scatter(k_x_peaks, k_y_peaks, s=30, color='lime', alpha=0.9, marker='o', edgecolors='black', linewidths=0.5, label="Peaks")
            
            if np.max(power[k_x_real**2 + k_y_real**2 > 1e-5]) > 0:
                k_x_max = k_x_real[max_idx]
                k_y_max = k_y_real[max_idx]
                axs[1].scatter(k_x_max, k_y_max, s=50, color='cyan', alpha=0.9, marker='o', edgecolors='black', linewidths=0.5, label="Max Peak")
                axs[1].set_xlim(-r_fundamental * 5.5, r_fundamental * 5.5)
                axs[1].set_ylim(-r_fundamental * 5.5, r_fundamental * 5.5)
        
        # 3rd Subplot: Angular Power Spectrum
        angles_deg = np.degrees(np.mod(angles, 2 * np.pi))
        hist, bin_edges = np.histogram(angles_deg, bins=180, range=(0, 360), weights=weights)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        axs[2].plot(bin_centers, hist, color='purple', linewidth=2)
        axs[2].fill_between(bin_centers, hist, color='purple', alpha=0.3)
        axs[2].set_title(f"Angular Power Spectrum\nClassified: {geometry}")
        axs[2].set_xlim(0, 360)
        axs[2].set_xticks([0, 90, 180, 270, 360])
        axs[2].set_xlabel("Angle (Degrees)")
        axs[2].set_ylabel("Integrated Power")
        
        textstr = f"$C_2$ (Spiral): {c2:.3f}\n$C_4$ (Square): {c4:.3f}\n$C_6$ (Hex): {c6:.3f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        axs[2].text(0.05, 0.95, textstr, transform=axs[2].transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
    plt.tight_layout()
    if outpath_override is not None:
        outpath = outpath_override
        os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    else:
        os.makedirs("output/plots/llg", exist_ok=True)
        outpath = f"output/plots/llg/FFT_{phase_name}.png"
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def analyze_state_real_space(spins, ax=1.0, ay=1.0, phase_name="Unknown", plot_fft=False, outpath_override=None, 
                             sigma=1.5, anisotropy_threshold=0.15, d_cut_multiplier=1.35, **kwargs):
    """
    Computes Topological Charge (Q) and categorizes phase symmetry ENTIRELY IN REAL SPACE.
    
    No Fourier transforms or reciprocal angular harmonics (C_n) are used. Instead:
    1. Uniform vs Non-Uniform: Evaluated via spin Sz variance.
    2. 1D vs 2D Ordering (SP): Evaluated via the out-of-plane magnetization n_z gradient structure tensor.
       A state is a 1D Spiral if the structure tensor anisotropy alpha > anisotropy_threshold (0.15).
    3. 4-fold vs 6-fold Symmetry (SC vs SkX): Evaluated by identifying periodic skyrmion cores (local minima)
       and calculating the local bond-orientational order parameters (psi_4 and psi_6) for nearest neighbors
       within d_cut = d_cut_multiplier * d_avg_1st.
    """
    # Enhanced spatial smoothing to filter out thermal high-frequency fluctuations
    n_smoothed = scipy.ndimage.gaussian_filter(spins, sigma=[sigma, sigma, 0], mode='wrap')
    
    # Re-normalize after smoothing
    norm = np.linalg.norm(n_smoothed, axis=-1, keepdims=True)
    n = n_smoothed / np.where(norm > 1e-12, norm, 1.0)
    
    n_right = np.roll(n, -1, axis=0)
    n_left = np.roll(n, 1, axis=0)
    n_up = np.roll(n, -1, axis=1)
    n_down = np.roll(n, 1, axis=1)
    
    dn_dx = (n_right - n_left) / (2.0 * ax)
    dn_dy = (n_up - n_down) / (2.0 * ay)
    
    # 1. Topological Charge (Q)
    charge_density = np.sum(n * np.cross(dn_dx, dn_dy), axis=-1) * ax * ay
    Q = np.sum(charge_density) / (4.0 * np.pi)
    
    # 2. Magnetization and Uniformity
    Mz = np.mean(spins[:, :, 2])
    n_z = n[:, :, 2]
    is_uniform = np.var(spins[:, :, 2]) < 1e-3
    
    # 3. Structure Tensor Analysis (for 1D vs 2D detection)
    nz_right = np.roll(n_z, -1, axis=0)
    nz_left = np.roll(n_z, 1, axis=0)
    nz_up = np.roll(n_z, -1, axis=1)
    nz_down = np.roll(n_z, 1, axis=1)
    
    g_x = (nz_right - nz_left) / (2.0 * ax)
    g_y = (nz_up - nz_down) / (2.0 * ay)
    
    J_xx = np.mean(g_x**2)
    J_yy = np.mean(g_y**2)
    J_xy = np.mean(g_x * g_y)
    
    trace = J_xx + J_yy
    det = J_xx * J_yy - J_xy**2
    
    sqrt_term = np.sqrt(np.maximum((trace / 2.0)**2 - det, 0.0))
    lambda1 = trace / 2.0 + sqrt_term
    lambda2 = trace / 2.0 - sqrt_term
    
    if lambda1 + lambda2 > 1e-12:
        anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2)
    else:
        anisotropy = 0.0
        
    # Default variables
    geometry = "Uniform (FM)"
    classified_state = "FM"
    psi4 = 0.0
    psi6 = 0.0
    coords = np.empty((0, 2))
    dist_copy = np.empty((0, 0))
    dp = np.empty((0, 0, 2))
    d_cut = 0.0
    
    if not is_uniform:
        if anisotropy > anisotropy_threshold:
            geometry = "1D Spiral"
            classified_state = "SP"
        else:
            # 4. Core Detection: Periodic Relative Minima
            min_filt = scipy.ndimage.minimum_filter(n_z, size=3, mode='wrap')
            is_min = (n_z == min_filt)
            mean_nz = np.mean(n_z)
            std_nz = np.std(n_z)
            
            # Find cores whose magnetization is lower than the mean minus a fraction of standard deviation
            core_mask = is_min & (n_z < mean_nz - 0.2 * std_nz)
            coords = np.argwhere(core_mask)
            N_cores = len(coords)
            
            if N_cores >= 2:
                L_x, L_y = n_z.shape
                box_x = L_x * ax
                box_y = L_y * ay
                positions = coords * np.array([ax, ay])
                
                # Pairwise periodic difference vectors
                dp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                dp[:, :, 0] = dp[:, :, 0] - np.round(dp[:, :, 0] / box_x) * box_x
                dp[:, :, 1] = dp[:, :, 1] - np.round(dp[:, :, 1] / box_y) * box_y
                
                distances = np.linalg.norm(dp, axis=-1)
                dist_copy = distances.copy()
                np.fill_diagonal(dist_copy, np.inf)
                
                first_nn_distances = np.min(dist_copy, axis=1)
                d_avg_1st = np.mean(first_nn_distances)
                d_cut = d_cut_multiplier * d_avg_1st
                
                psi4_list = []
                psi6_list = []
                
                for i in range(N_cores):
                    neighbor_indices = np.where(dist_copy[i] < d_cut)[0]
                    N_i = len(neighbor_indices)
                    if N_i > 0:
                        dx = dp[i, neighbor_indices, 0]
                        dy = dp[i, neighbor_indices, 1]
                        angles_i = np.arctan2(dy, dx)
                        
                        psi4_i = np.abs(np.mean(np.exp(1j * 4 * angles_i)))
                        psi6_i = np.abs(np.mean(np.exp(1j * 6 * angles_i)))
                        
                        psi4_list.append(psi4_i)
                        psi6_list.append(psi6_i)
                        
                if len(psi4_list) > 0:
                    psi4 = np.mean(psi4_list)
                    psi6 = np.mean(psi6_list)
                    
                if psi6 > psi4:
                    geometry = "2D Hexagonal"
                    classified_state = "SkX"
                else:
                    geometry = "2D Square"
                    classified_state = "SC"
            else:
                # Fallback if fewer than 2 cores are detected (isolated skyrmions/dilute phase)
                if abs(Q) > 0.5:
                    geometry = "2D Hexagonal"
                    classified_state = "SkX"
                else:
                    geometry = "2D Square"
                    classified_state = "SC"
                    
    topology = "Skyrmionic" if (not is_uniform and abs(Q) > 0.5) else "Trivial"
    
    if plot_fft:  # Generate real space visual diagnostics when plot_fft is True
        _plot_analysis_real_space(n_z, coords, dp, dist_copy, d_cut, is_uniform, anisotropy, 
                                  psi4, psi6, geometry, phase_name, outpath_override, ax, ay)
        
    return {
        "Q": Q, "Mz": Mz, "geometry": geometry, "topology": topology, 
        "peaks": len(coords), "is_uniform": is_uniform, "classified_state": classified_state,
        "c2": anisotropy, "c4": psi4, "c6": psi6
    }


def _plot_analysis_real_space(n_z, cores, dp, dist_copy, d_cut, is_uniform, anisotropy, psi4, psi6,
                              geometry, phase_name, outpath_override, ax, ay):
    """Internal helper to render real-space diagnostic plots of cores and bond order parameters."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    L_x, L_y = n_z.shape
    extent_real = [0, L_x * ax, 0, L_y * ay]
    
    # Subplot 0: n_z raw real-space state
    im0 = axs[0].imshow(n_z.T, cmap='bwr', extent=extent_real, origin='lower', vmin=-1.0, vmax=1.0)
    axs[0].set_title(f"Real Space ($n_z$) - {phase_name}")
    axs[0].set_xlabel("X (Physical)")
    axs[0].set_ylabel("Y (Physical)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Subplot 1: Detected Cores
    axs[1].imshow(n_z.T, cmap='bwr', extent=extent_real, origin='lower', vmin=-1.0, vmax=1.0, alpha=0.6)
    if len(cores) > 0:
        phys_cores = cores * np.array([ax, ay])
        axs[1].scatter(phys_cores[:, 0], phys_cores[:, 1], color='lime', edgecolor='black', s=50, zorder=3, label="Cores")
        axs[1].legend(loc='upper right')
    axs[1].set_title(f"Core Detection\n{len(cores)} Cores Detected")
    axs[1].set_xlabel("X (Physical)")
    axs[1].set_ylabel("Y (Physical)")
    
    # Subplot 2: Periodic Bond Network
    axs[2].set_facecolor('#f0f0f5')
    if len(cores) >= 2:
        phys_cores = cores * np.array([ax, ay])
        N_cores = len(cores)
        for i in range(N_cores):
            neighbor_indices = np.where(dist_copy[i] < d_cut)[0]
            for j in neighbor_indices:
                p_start = phys_cores[i]
                p_end = phys_cores[i] + dp[i, j]
                axs[2].plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color='purple', alpha=0.5, linewidth=1.5, zorder=1)
        axs[2].scatter(phys_cores[:, 0], phys_cores[:, 1], color='red', edgecolor='black', s=60, zorder=3)
    else:
        axs[2].text(0.5, 0.5, "Insufficient Cores\nfor Bond Network", ha='center', va='center', fontsize=12)
        
    axs[2].set_xlim(0, L_x * ax)
    axs[2].set_ylim(0, L_y * ay)
    axs[2].set_title(f"Bond Orientational Order\nClassified: {geometry}")
    axs[2].set_xlabel("X (Physical)")
    axs[2].set_ylabel("Y (Physical)")
    
    # Add textual info box
    textstr = (f"Anisotropy: {anisotropy:.3f}\n"
               f"Cores: {len(cores)}\n"
               f"$\psi_4$ (Square): {psi4:.3f}\n"
               f"$\psi_6$ (Hex): {psi6:.3f}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    axs[2].text(0.05, 0.95, textstr, transform=axs[2].transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
                
    plt.tight_layout()
    if outpath_override is not None:
        outpath = outpath_override
        os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    else:
        os.makedirs("output/plots/mc", exist_ok=True)
        outpath = f"output/plots/mc/RS_{phase_name}.png"
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

