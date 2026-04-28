import numpy as np
import numba as nb
import argparse
import os

# ---------------------------------------------------------
# Part A: Ansatz Generators
# ---------------------------------------------------------

def init_SkX(L, Q=1, gamma=0.0):
    """
    Hexagonal 3Q Skyrmion Lattice (SkX) Ansatz
    A proper vector superposition of three cycloids to form an exact hexagonal lattice.
    Mapped onto physical coordinates [-L_x/2, L_x/2] and [-L_y/2, L_y/2].
    Returns spins, ax, ay
    """
    spins = np.zeros((L, L, 3))
    
    ax = np.pi / L
    ay = (np.pi / np.sqrt(3)) / L
    
    # We use q=4.0 so that exactly two complementary skyrmions precisely fit 
    # the periodic rectangular unit cell (L_x = pi, L_y = pi/sqrt(3)).
    # Dynamic scaling (ax, ay) during LLG will natively expand this to the optimal physical period.
    q_ansatz = 4.0 
    
    # Base offset to ensure SkX background matches the FM -z state for stability
    # Superposition varies from -1.5 to +3.0. A -1.5 offset makes it vary from -3.0 to +1.5.
    M0 = -1.5 
    
    for i in range(L):
        for j in range(L):
            x = (i - L/2 + 0.5) * ax
            y = (j - L/2 + 0.5) * ay
            
            # The three k-vectors for the hexagonal Brillouin zone
            W1 = q_ansatz * x
            W2 = q_ansatz * (-0.5 * x + (np.sqrt(3)/2) * y)
            W3 = q_ansatz * (-0.5 * x - (np.sqrt(3)/2) * y)
            
            nz = M0 + np.cos(W1) + np.cos(W2) + np.cos(W3)
            
            # Néel skyrmion: in-plane components are parallel to the q-vectors. 
            # Subtracted mathematically so DMI chirality stabilizes negatively (matches our q=-2 fix for SP).
            sign = -1.0 
            nx = sign * (np.sin(W1)*1.0 + np.sin(W2)*(-0.5) + np.sin(W3)*(-0.5))
            ny = sign * (np.sin(W1)*0.0 + np.sin(W2)*(np.sqrt(3)/2) + np.sin(W3)*(-np.sqrt(3)/2))
            
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            if norm == 0:
                spins[i, j, 2] = -1.0
            else:
                spins[i, j, 0] = nx / norm
                spins[i, j, 1] = ny / norm
                spins[i, j, 2] = nz / norm
                
    return spins, ax, ay

def init_SP(L):
    """
    Spiral Phase (SP) Ansatz
    Mapped onto physical coordinates [0, L_x] and [0, L_y].
    Returns spins, ax, ay
    """
    spins = np.zeros((L, L, 3))
    
    ax = np.pi / L
    ay = np.pi / L
    
    # We use q=2.0 since the cos/sin phase arrangement naturally yields negative DMI
    q = 2.0 
    
    for i in range(L):
        for j in range(L):
            x = i * ax
            spins[i, j, 0] = np.cos(q * x)
            spins[i, j, 1] = 0.0
            spins[i, j, 2] = np.sin(q * x)
            
    return spins, ax, ay

def init_SC(L):
    """
    Square Cell (SC) Vortex-Antivortex Phase Ansatz
    Mapped onto physical coordinates [0, L_x] and [0, L_y].
    Returns spins, ax, ay
    """
    spins = np.zeros((L, L, 3))
    
    ax = (np.pi / 2.0) / L
    ay = (np.pi / 2.0) / L
    
    # We use q=4.0 (+4.0) because the user's specific sin/cos permutation natively yields negative DMI
    q = 4.0 
    
    for i in range(L):
        for j in range(L):
            x = i * ax
            y = j * ay
            
            # Wave 1: Propagates along x
            n1_x = np.cos(q * x)
            n1_y = 0.0
            n1_z = np.sin(q * x)
            
            # Wave 2: Propagates along y
            n2_x = 0.0
            n2_y = np.cos(q * y)
            n2_z = np.sin(q * y)
            
            n_sum_x = n1_x + n2_x
            n_sum_y = n1_y + n2_y
            n_sum_z = n1_z + n2_z
            
            norm = np.sqrt(n_sum_x**2 + n_sum_y**2 + n_sum_z**2)
            
            # Normalization (Crucial Step)
            if norm == 0:
                spins[i, j, 2] = -1.0 # Fallback aligned to negative H field preference
            else:
                spins[i, j, 0] = n_sum_x / norm
                spins[i, j, 1] = n_sum_y / norm
                spins[i, j, 2] = n_sum_z / norm
                
    return spins, ax, ay

def load_ansatz(filepath, L):
    """Load MC or LLG output (npy or npz) as an ansatz."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}")
        
    ax, ay = 1.0, 1.0
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        spins = data['spins']
        if 'ax' in data: ax = float(data['ax'])
        if 'ay' in data: ay = float(data['ay'])
    else:
        spins = np.load(filepath)
        
    if spins.shape != (L, L, 3):
        raise ValueError(f"Shape mismatch: {spins.shape} vs {(L, L, 3)}")
    return spins, ax, ay

# ---------------------------------------------------------
# Part B & C: Energy, Effective Field, and LLG step
# ---------------------------------------------------------

@nb.njit
def relax_phase_numba(spins, L, H_scaled, A_scaled, max_steps=50000, tol=1e-7, ax_in=1.0, ay_in=1.0, prev_f_in=0.0, max_dt=0.05, cfl_factor=0.25, global_step_start=0):
    """
    Perform the full overdamped LLG integration natively in Numba using Heun (RK2) method.
    This avoids Python overhead and memory allocations at every step, making it ~100x faster.
    """
    ax = ax_in
    ay = ay_in
    prev_f = prev_f_in
    dt = max_dt 
    
    # Pre-allocate buffers for ping-pong swapping and predictor state
    spins_current = spins.copy()
    spins_next = np.empty_like(spins)
    spins_pred = np.empty_like(spins)
    ndot_current = np.empty_like(spins)
    
    f_tot = 0.0
    
    for step in range(max_steps):
        # Dynamic timestep bounded by Von Neumann stability analysis (CFL condition)
        dt = min(max_dt, cfl_factor * min(ax, ay)**2)
        
        # Energy accumulators (unscaled)
        E_ex_x = 0.0
        E_ex_y = 0.0
        E_dmi_x = 0.0
        E_dmi_y = 0.0
        E_z = 0.0
        E_a = 0.0
        
        # --- PASS 1: Predictor Step ---
        for i in range(L):
            for j in range(L):
                # Periodic neighbors
                i_right = (i + 1) % L
                i_left = (i - 1 + L) % L
                j_up = (j + 1) % L
                j_down = (j - 1 + L) % L
                
                # Unroll vectors heavily to optimize C-compilation
                n_x = spins_current[i, j, 0]
                n_y = spins_current[i, j, 1]
                n_z = spins_current[i, j, 2]
                
                n_right_x, n_right_y, n_right_z = spins_current[i_right, j, 0], spins_current[i_right, j, 1], spins_current[i_right, j, 2]
                n_left_x, n_left_y, n_left_z = spins_current[i_left, j, 0], spins_current[i_left, j, 1], spins_current[i_left, j, 2]
                
                n_up_x, n_up_y, n_up_z = spins_current[i, j_up, 0], spins_current[i, j_up, 1], spins_current[i, j_up, 2]
                n_down_x, n_down_y, n_down_z = spins_current[i, j_down, 0], spins_current[i, j_down, 1], spins_current[i, j_down, 2]
                
                # --- Energy Accumulation ---
                E_ex_x += 0.5 * ((n_right_x - n_x)**2 + (n_right_y - n_y)**2 + (n_right_z - n_z)**2)
                E_ex_y += 0.5 * ((n_up_x - n_x)**2 + (n_up_y - n_y)**2 + (n_up_z - n_z)**2)
                
                E_dmi_x += (n_z * (n_right_x - n_x) - n_x * (n_right_z - n_z))
                E_dmi_y += (n_z * (n_up_y - n_y) - n_y * (n_up_z - n_z))
                
                E_z += H_scaled * n_z
                E_a += A_scaled * n_z**2
                
                # --- Effective Field Calculation ---
                H_x = (n_right_x + n_left_x - 2*n_x)/(ax**2) + (n_up_x + n_down_x - 2*n_x)/(ay**2)
                H_y = (n_right_y + n_left_y - 2*n_y)/(ax**2) + (n_up_y + n_down_y - 2*n_y)/(ay**2)
                H_z = (n_right_z + n_left_z - 2*n_z)/(ax**2) + (n_up_z + n_down_z - 2*n_z)/(ay**2)
                
                # DMI Effective Field (Factor of 2 removed from denominator via analytical Integration By Parts)
                H_x += (n_right_z - n_left_z) / ax
                H_y += (n_up_z - n_down_z) / ay
                H_z += - (n_right_x - n_left_x) / ax - (n_up_y - n_down_y) / ay
                
                H_z -= H_scaled
                H_z -= 2 * A_scaled * n_z
                
                # --- LLG Derivative (Predictor) ---
                dot_val = n_x*H_x + n_y*H_y + n_z*H_z
                ndot_x = H_x - dot_val * n_x
                ndot_y = H_y - dot_val * n_y
                ndot_z = H_z - dot_val * n_z
                
                # Save derivative for corrector
                ndot_current[i, j, 0] = ndot_x
                ndot_current[i, j, 1] = ndot_y
                ndot_current[i, j, 2] = ndot_z
                
                # Predictor Step
                n_pred_x = n_x + dt * ndot_x
                n_pred_y = n_y + dt * ndot_y
                n_pred_z = n_z + dt * ndot_z
                
                # Normalize predictor
                norm_pred = np.sqrt(n_pred_x**2 + n_pred_y**2 + n_pred_z**2)
                spins_pred[i, j, 0] = n_pred_x / norm_pred
                spins_pred[i, j, 1] = n_pred_y / norm_pred
                spins_pred[i, j, 2] = n_pred_z / norm_pred

        # Average energies per spin
        L2 = L*L
        E_ex_x /= L2
        E_ex_y /= L2
        E_dmi_x /= L2
        E_dmi_y /= L2
        E_z /= L2
        E_a /= L2
        
        # Total Scaled Energy Density
        f_tot = (E_ex_x / ax**2) + (E_ex_y / ay**2) + (E_dmi_x / ax) + (E_dmi_y / ay) + E_z + E_a
        
        # --- PASS 2: Corrector Step ---
        for i in range(L):
            for j in range(L):
                # Periodic neighbors
                i_right = (i + 1) % L
                i_left = (i - 1 + L) % L
                j_up = (j + 1) % L
                j_down = (j - 1 + L) % L
                
                # Predictor spin states
                n_x = spins_pred[i, j, 0]
                n_y = spins_pred[i, j, 1]
                n_z = spins_pred[i, j, 2]
                
                n_right_x, n_right_y, n_right_z = spins_pred[i_right, j, 0], spins_pred[i_right, j, 1], spins_pred[i_right, j, 2]
                n_left_x, n_left_y, n_left_z = spins_pred[i_left, j, 0], spins_pred[i_left, j, 1], spins_pred[i_left, j, 2]
                
                n_up_x, n_up_y, n_up_z = spins_pred[i, j_up, 0], spins_pred[i, j_up, 1], spins_pred[i, j_up, 2]
                n_down_x, n_down_y, n_down_z = spins_pred[i, j_down, 0], spins_pred[i, j_down, 1], spins_pred[i, j_down, 2]
                
                # --- Effective Field Calculation (Corrector) ---
                H_x = (n_right_x + n_left_x - 2*n_x)/(ax**2) + (n_up_x + n_down_x - 2*n_x)/(ay**2)
                H_y = (n_right_y + n_left_y - 2*n_y)/(ax**2) + (n_up_y + n_down_y - 2*n_y)/(ay**2)
                H_z = (n_right_z + n_left_z - 2*n_z)/(ax**2) + (n_up_z + n_down_z - 2*n_z)/(ay**2)
                
                H_x += (n_right_z - n_left_z) / ax
                H_y += (n_up_z - n_down_z) / ay
                H_z += - (n_right_x - n_left_x) / ax - (n_up_y - n_down_y) / ay
                
                H_z -= H_scaled
                H_z -= 2 * A_scaled * n_z
                
                # --- LLG Derivative (Corrector) ---
                dot_val = n_x*H_x + n_y*H_y + n_z*H_z
                ndot_x_pred = H_x - dot_val * n_x
                ndot_y_pred = H_y - dot_val * n_y
                ndot_z_pred = H_z - dot_val * n_z
                
                # --- Corrector Step ---
                n_x_orig = spins_current[i, j, 0]
                n_y_orig = spins_current[i, j, 1]
                n_z_orig = spins_current[i, j, 2]
                
                ndot_x_orig = ndot_current[i, j, 0]
                ndot_y_orig = ndot_current[i, j, 1]
                ndot_z_orig = ndot_current[i, j, 2]
                
                ndot_x_avg = 0.5 * (ndot_x_orig + ndot_x_pred)
                ndot_y_avg = 0.5 * (ndot_y_orig + ndot_y_pred)
                ndot_z_avg = 0.5 * (ndot_z_orig + ndot_z_pred)
                
                n_new_x = n_x_orig + dt * ndot_x_avg
                n_new_y = n_y_orig + dt * ndot_y_avg
                n_new_z = n_z_orig + dt * ndot_z_avg
                
                # Strict Re-normalization
                norm = np.sqrt(n_new_x**2 + n_new_y**2 + n_new_z**2)
                spins_next[i, j, 0] = n_new_x / norm
                spins_next[i, j, 1] = n_new_y / norm
                spins_next[i, j, 2] = n_new_z / norm

        # Dynamic Scaling Strategy: Use a low-pass filter (EMA) to dampen vicious cycles
        alpha_scale = 0.01
        if abs(E_dmi_x) > 1e-12 and abs(E_ex_x) > 1e-12:
            # We strictly enforce the positive root of the energy minimization polynomial
            target_ax = 2.0 * E_ex_x / abs(E_dmi_x)
            ax = (1.0 - alpha_scale) * ax + alpha_scale * target_ax
        if abs(E_dmi_y) > 1e-12 and abs(E_ex_y) > 1e-12:
            target_ay = 2.0 * E_ex_y / abs(E_dmi_y)
            ay = (1.0 - alpha_scale) * ay + alpha_scale * target_ay

        # Enforce isotropy for 1D states to prevent extreme aspect ratios 
        # which break the continuum Laplacian and ruin SDE condition numbers
        if abs(E_dmi_y) <= 1e-10 and abs(E_ex_y) <= 1e-10 and abs(E_dmi_x) > 1e-10:
            ay = ax
        if abs(E_dmi_x) <= 1e-10 and abs(E_ex_x) <= 1e-10 and abs(E_dmi_y) > 1e-10:
            ax = ay
            
        # Optional clamping
        if ax <= 0: ax = ax_in
        if ay <= 0: ay = ay_in

        # Check convergence
        if (global_step_start + step) > 3000 and abs(f_tot - prev_f) < tol:
            # We must break here and transfer the output
            spins_current[:] = spins_next[:]
            break
            
        prev_f = f_tot
        
        # Swap buffers for the next timestep (avoids np.zeros_like allocation)
        temp = spins_current
        spins_current = spins_next
        spins_next = temp
        
    return spins_current, f_tot, ax, ay, step

# ---------------------------------------------------------
# Part D & E: Execution Wrapper & Analysis
# ---------------------------------------------------------

def analyze_state(spins, ax, ay, phase_name="Unknown", plot_fft=False):
    """
    Computes Topological Charge (Q) and categorizes phase symmetry 
    by checking physical gradients and Fast Fourier Transform (FFT) Bragg peaks.
    Properly maps continuous scale (ax, ay) directly into reciprocal resolution.
    """
    import scipy.ndimage
    # Light Gaussian blur to filter out finite-temperature high-frequency magnons (noise)
    # This stabilizes both the topological charge derivatives and the FFT peak detection.
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
    
    # 1. Topological Charge (Q) via Cross Product continuous projection
    charge_density = np.sum(n * np.cross(dn_dx, dn_dy), axis=-1) * ax * ay
    Q = np.sum(charge_density) / (4.0 * np.pi)
    
    # 2. Geometry via FFT Analysis
    Mz = np.mean(n[:, :, 2])
    n_z = n[:, :, 2]
    
    # Remove DC (Uniform) component to isolate structural periodicity
    fft_z = np.fft.fftshift(np.fft.fft2(n_z - Mz))
    power = np.abs(fft_z)**2
    
    # Explicitly enforce DC bin = 0.0 to prevent bleed 
    L_x, L_y = n_z.shape
    power[L_x//2, L_y//2] = 0.0
    
    max_power = np.max(power)
    
    # Use spatial variance of the smoothed spin field to reliably detect Uniform/FM states
    # Structural phases (SkX, SC, SP) have macroscopic variance > 0.05. Thermal noise variance is < 0.01.
    is_uniform = np.var(n[:, :, 2]) < 0.01
    
    num_peaks = 0
    mask = None
    geometry = "Uniform (FM)"
    
    if not is_uniform:
        import scipy.ndimage
        # Set max_power threshold really low to capture higher harmonics
        mask = power > (max_power * 0.4)
        # Contiguous cluster linking. Use 4-connectivity (2, 1) to prevent 
        # distinct diagonal Bragg peaks from merging together on tight grid scales.
        s = scipy.ndimage.generate_binary_structure(2, 1)
        labels, num_peaks = scipy.ndimage.label(mask, structure=s)
        
        # Determine if peaks are collinear (1D) or spread out (2D)
        if num_peaks > 0:
            peak_centers = scipy.ndimage.center_of_mass(mask, labels, np.arange(1, num_peaks + 1))
            # peak_centers is a list of (y, x) tuples
            dy = np.array([p[0] for p in peak_centers]) - (L_x // 2)
            dx = np.array([p[1] for p in peak_centers]) - (L_y // 2)
            
            # Calculate angles, double them (to ignore pi phase shifts), and find phase coherence R
            angles = np.arctan2(dy, dx)
            v_x = np.cos(2 * angles)
            v_y = np.sin(2 * angles)
            R = np.sqrt(np.mean(v_x)**2 + np.mean(v_y)**2)
            
            # If num_peaks is 1, it might be a contiguous ring around the center (common for tight square lattices)
            if num_peaks == 1:
                coords = np.argwhere(mask)
                height = np.max(coords[:, 0]) - np.min(coords[:, 0])
                width = np.max(coords[:, 1]) - np.min(coords[:, 1])
                # Distinguish a 1D line from a 2D ring based on bounding box aspect ratio
                is_collinear = (height < 2) or (width < 2) or (max(height, width) / max(min(height, width), 1) > 2.0)
            else:
                is_collinear = R > 0.9
        else:
            is_collinear = False
            
        # Decouple topology and geometry to handle phases that defect into Skyrmionic topologies (like SC)
        if num_peaks == 0:
            geometry = "Uniform (FM)"
        elif is_collinear:
            if num_peaks % 2 == 0 or num_peaks == 1:
                geometry = "1D Spiral"
            else:
                geometry = "Distorted Lattice"
        else:
            if num_peaks % 6 == 0:
                geometry = "2D Hexagonal"
            elif num_peaks % 4 == 0 or num_peaks in [1, 8, 9, 12]:
                geometry = "2D Square"
            else:
                geometry = "Distorted Lattice"
                
    # 3. Topology Classification
    topology = "Skyrmionic" if (not is_uniform and abs(Q) > 0.5) else "Trivial"
    
    # 4. Final Definitive State Mapping
    classified_state = "FM"
    if not is_uniform:
        if geometry == "2D Hexagonal" and topology == "Skyrmionic":
            classified_state = "SkX"
        elif geometry == "2D Square":
            classified_state = "SC"
        elif geometry == "1D Spiral" and abs(Q) < 0.5:
            classified_state = "SP"
        else:
            classified_state = "Unknown Phase"
            
    if plot_fft:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        L_x, L_y = n_z.shape
        extent_real = [-0.5 * L_x * ax, 0.5 * L_x * ax, -0.5 * L_y * ay, 0.5 * L_y * ay]
        
        axs[0].imshow(n_z.T, cmap='bwr', extent=extent_real, origin='lower')
        axs[0].set_title(f"Real Space ($n_z$) - {phase_name}")
        axs[0].set_xlabel("X (Physical)")
        axs[0].set_ylabel("Y (Physical)")
        
        if is_uniform:
            axs[1].text(0.5, 0.5, "Uniform State\n(No Peaks)", ha='center', va='center')
            axs[1].axis('off')
        else:
            # Calculate physical k-space bounds
            k_max_x = np.pi / ax
            k_max_y = np.pi / ay
            extent_k = [-k_max_x, k_max_x, -k_max_y, k_max_y]
            
            im = axs[1].imshow(np.log10(power.T + 1e-12), cmap='magma', extent=extent_k, origin='lower', interpolation='bicubic')
            axs[1].set_title(f"Reciprocal Power Spectrum (Log)\n{num_peaks} Fundamental Peaks")
            axs[1].set_xlabel(r"$k_x$")
            axs[1].set_ylabel(r"$k_y$")
            axs[1].contour(mask.T, levels=[0.5], colors='cyan', linewidths=2, extent=extent_k)
            
        plt.tight_layout()
        os.makedirs("output/LLG/Graphs", exist_ok=True)
        outpath = f"output/LLG/Graphs/FFT_{phase_name}.png"
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
        
    return {
        "Q": Q, 
        "Mz": Mz, 
        "geometry": geometry, 
        "topology": topology, 
        "peaks": num_peaks, 
        "is_uniform": is_uniform,
        "classified_state": classified_state
    }

def relax_phase(spins, L, H_scaled, A_scaled, phase_name, ax_in=1.0, ay_in=1.0, max_steps=50000, tol=1e-7, live_plot=False, live_mode="quiver", max_dt=0.05, cfl_factor=0.25, visualize_scaling=False):
    """
    Relax the given spin configuration using LLG and dynamic scaling.
    This is now a wrapper around the ultra-fast Numba integrator.
    """
    ax, ay = ax_in, ay_in
    prev_f = 0.0
    steps_done = 0
    
    if live_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax_plot = plt.subplots(figsize=(8,8))
        
        if live_mode == "quiver":
            tiles_x, tiles_y = 1, 1 # Change here number of tiles that are live plotted
            tiled_spins = np.tile(spins, (tiles_x, tiles_y, 1))
            L_x, L_y = tiled_spins.shape[0], tiled_spins.shape[1]
            X_base, Y_base = np.meshgrid(np.arange(L_x), np.arange(L_y))
            X, Y = X_base * ax, Y_base * ay
            
            U = tiled_spins[:, :, 0].T
            V = tiled_spins[:, :, 1].T
            Sz = tiled_spins[:, :, 2].T
            
            q = ax_plot.quiver(X, Y, U, V, Sz, cmap="coolwarm", pivot='mid', scale=max(L_x, L_y)*0.8, width=0.005)
            q.set_clim(-1, 1)
            
            rects = []
            for i in range(tiles_x):
                for j in range(tiles_y):
                    rect = plt.Rectangle(((-0.5 + i*L)*ax, (-0.5 + j*L)*ay), L*ax, L*ay, 
                                       fill=False, edgecolor='black', linestyle='--', linewidth=1.5, alpha=0.5)
                    ax_plot.add_patch(rect)
                    rects.append((i, j, rect))
                    
            ax_plot.set_xlim(-ax, L_x * ax)
            ax_plot.set_ylim(-ay, L_y * ay)
            ax_plot.set_aspect('equal')
        else:
            im = ax_plot.imshow(spins[:, :, 2], vmin=-1, vmax=1, cmap='bwr', origin='lower', extent=[0, L*ax, 0, L*ay])
            ax_plot.set_aspect('equal')

        ax_plot.set_title(f"Relaxing {phase_name}...")
        fig.tight_layout()
        plt.show()
    
    chunk = 10 if live_plot else max_steps
    
    while steps_done < max_steps:
        # Numba executes cleanly in small chunks so we can visualize intermediate states
        spins_final, f_tot, ax, ay, steps_taken = relax_phase_numba(
            spins, L, H_scaled, A_scaled, chunk, tol, ax, ay, prev_f, max_dt, cfl_factor, global_step_start=steps_done
        )
        
        # The true number of steps executed is steps_taken + 1 because the loop is 0-indexed and returns `step`
        steps_done += (steps_taken + 1)
        prev_f = f_tot
        spins = spins_final
        
        if live_plot:
            if live_mode == "quiver":
                tiled_spins = np.tile(spins, (tiles_x, tiles_y, 1))
                U = tiled_spins[:, :, 0].T
                V = tiled_spins[:, :, 1].T
                Sz = tiled_spins[:, :, 2].T
                q.set_UVC(U, V, Sz)
                
                # Always update positions for real physical scaling
                pts = np.column_stack((X_base.flatten() * ax, Y_base.flatten() * ay))
                q.set_offsets(pts)
                
                for (i, j, rect) in rects:
                    rect.set_xy(((-0.5 + i*L)*ax, (-0.5 + j*L)*ay))
                    rect.set_width(L * ax)
                    rect.set_height(L * ay)
                    
                ax_plot.set_xlim(-ax, L_x * ax)
                ax_plot.set_ylim(-ay, L_y * ay)
            else:
                im.set_data(spins[:, :, 2])
                im.set_extent([0, L*ax, 0, L*ay])
                ax_plot.set_xlim(0, L*ax)
                ax_plot.set_ylim(0, L*ay)
                
            ax_plot.set_title(f"[{phase_name}] Step {steps_done} | f={f_tot:.4f} | ax={ax:.2f}, ay={ay:.2f}")
            plt.pause(0.02)
            
        if steps_taken + 1 < chunk:
            break
            
    if live_plot:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.close(fig)
    
    print(f"[{phase_name}] Relaxed in {steps_done} steps. Energy Density: {f_tot:.5f} (ax={ax:.3f}, ay={ay:.3f})")
    return spins, f_tot, ax, ay

def get_FM_energy(H_scaled, A_scaled):
    """Calculate the exact analytical energy of the FM state."""
    # Aligned FM (nz = 1 or -1)
    # The energy is E = A_s n_z^2 + H n_z
    e_aligned = A_scaled + H_scaled # For nz = +1
    e_anti_aligned = A_scaled - H_scaled # For nz = -1
    
    # Tilted FM (occurs when |H| < 2|A_s| and A_s > 0 usually)
    # nz = H / (2 A_s)
    e_tilted = float('inf')
    if A_scaled != 0:
        nz_tilted = H_scaled / (2 * A_scaled)
        if abs(nz_tilted) <= 1.0:
            e_tilted = - (H_scaled**2) / (4 * A_scaled)
            
    return min(e_aligned, e_anti_aligned, e_tilted)

def compare_phases(H_scaled=0.08, A_scaled=0.5, L=64, npy_file=None, plot_ansatz=False, live_plot=False, live_mode="quiver", max_dt=0.05, cfl_factor=0.25, visualize_scaling=False, plot_groundstate=False, save_outputs=True, plot_fft=False):
    """
    Main Execution: Tests SkX, SP, and FM to find the true numerical ground state.
    """
    print(f"--- Phase Stability Analysis H={H_scaled}, As={A_scaled} ---")
    results = {}
    
    # 1. Skyrmion Lattice
    print("Initializing SkX...")
    spins_skx, ax_skx, ay_skx = init_SkX(L)
    if save_outputs or plot_ansatz:
        np.savez("output/LLG/Ansatze/ansatz_SkX.npz", spins=spins_skx, ax=ax_skx, ay=ay_skx)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SkX Ansatz...")
            plot_periodic_structure("output/LLG/Ansatze/ansatz_SkX.npz", tiles_x=2, tiles_y=2, display_mode="quiver", ax=ax_skx, ay=ay_skx)
        except: pass
    spins_skx, f_skx, final_ax_skx, final_ay_skx = relax_phase(spins_skx, L, H_scaled, A_scaled, "SkX", ax_in=ax_skx, ay_in=ay_skx, live_plot=live_plot, live_mode=live_mode, max_dt=max_dt, cfl_factor=cfl_factor, visualize_scaling=visualize_scaling)
    stats_skx = analyze_state(spins_skx, final_ax_skx, final_ay_skx, phase_name="SkX", plot_fft=plot_fft)
    print(f"   -> Q={stats_skx['Q']:.2f} | Geo: {stats_skx['geometry']} ({stats_skx['peaks']} Peaks) -> Detected as: {stats_skx['classified_state']}")
    
    # 2. Square Cell 
    print("Initializing SC...")
    spins_sc, ax_sc, ay_sc = init_SC(L)
    if save_outputs or plot_ansatz:
        np.savez("output/LLG/Ansatze/ansatz_SC.npz", spins=spins_sc, ax=ax_sc, ay=ay_sc)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SC Ansatz...")
            plot_periodic_structure("output/LLG/Ansatze/ansatz_SC.npz", tiles_x=2, tiles_y=2, display_mode="quiver", ax=ax_sc, ay=ay_sc)
        except: pass
    spins_sc, f_sc, final_ax_sc, final_ay_sc = relax_phase(spins_sc, L, H_scaled, A_scaled, "SC", ax_in=ax_sc, ay_in=ay_sc, live_plot=live_plot, live_mode=live_mode, max_dt=max_dt, cfl_factor=cfl_factor, visualize_scaling=visualize_scaling)
    stats_sc = analyze_state(spins_sc, final_ax_sc, final_ay_sc, phase_name="SC", plot_fft=plot_fft)
    print(f"   -> Q={stats_sc['Q']:.2f} | Geo: {stats_sc['geometry']} ({stats_sc['peaks']} Peaks) -> Detected as: {stats_sc['classified_state']}")
    
    # 3. Spiral Phase
    print("Initializing SP...")
    spins_sp, ax_sp, ay_sp = init_SP(L)
    if save_outputs or plot_ansatz:
        np.savez("output/LLG/Ansatze/ansatz_SP.npz", spins=spins_sp, ax=ax_sp, ay=ay_sp)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SP Ansatz...")
            plot_periodic_structure("output/LLG/Ansatze/ansatz_SP.npz", tiles_x=2, tiles_y=2, display_mode="quiver", ax=ax_sp, ay=ay_sp)
        except: pass
    spins_sp, f_sp, final_ax_sp, final_ay_sp = relax_phase(spins_sp, L, H_scaled, A_scaled, "SP", ax_in=ax_sp, ay_in=ay_sp, live_plot=live_plot, live_mode=live_mode, max_dt=max_dt, cfl_factor=cfl_factor, visualize_scaling=visualize_scaling)
    stats_sp = analyze_state(spins_sp, final_ax_sp, final_ay_sp, phase_name="SP", plot_fft=plot_fft)
    print(f"   -> Q={stats_sp['Q']:.2f} | Geo: {stats_sp['geometry']} ({stats_sp['peaks']} Peaks) -> Detected as: {stats_sp['classified_state']}")
    
    # 4. Ferromagnetic
    f_fm = get_FM_energy(H_scaled, A_scaled)
    print(f"[FM] Analytical Energy Density: {f_fm:.5f}")
    
    # 5. Custom Ansatz (Optional)
    if npy_file and os.path.exists(npy_file):
        print(f"Initializing from custom file {npy_file}...")
        spins_cust, ax_cust, ay_cust = load_ansatz(npy_file, L)
        if plot_ansatz:
            try:
                from periodic_plotting import plot_periodic_structure
                print("Displaying Custom Ansatz...")
                plot_periodic_structure(npy_file, tiles_x=2, tiles_y=2, display_mode="quiver", ax=ax_cust, ay=ay_cust)
            except: pass
        spins_cust, f_cust, final_ax_cust, final_ay_cust = relax_phase(spins_cust, L, H_scaled, A_scaled, "Custom", ax_in=ax_cust, ay_in=ay_cust, live_plot=live_plot, live_mode=live_mode, max_dt=max_dt, cfl_factor=cfl_factor)
        stats_cust = analyze_state(spins_cust, final_ax_cust, final_ay_cust, phase_name="Custom", plot_fft=plot_fft)
    # -------------------------------------------------------------
    # Phase Classification and Energy Dynamic Mapping
    # -------------------------------------------------------------
    final_energies = {"FM": f_fm}  # FM limit is analytically guaranteed
    best_states = {} # Save winning spins mapped to true physical label
    
    candidates = [
        ("SkX", spins_skx, final_ax_skx, final_ay_skx, f_skx, stats_skx),
        ("SC", spins_sc, final_ax_sc, final_ay_sc, f_sc, stats_sc),
        ("SP", spins_sp, final_ax_sp, final_ay_sp, f_sp, stats_sp)
    ]
    if npy_file and os.path.exists(npy_file):
        candidates.append(("Custom", spins_cust, final_ax_cust, final_ay_cust, f_cust, stats_cust))
        
    print("\n--- Validation & Classification ---")
    for ansatz_name, spins_k, ax_k, ay_k, f_k, stats_k in candidates:
        actual_class = stats_k["classified_state"]
        
        # Grid Divergence Checks (Unphysical SDE numerical breakdown)
        diverged_grid = ax_k > 50.0 or ay_k > 50.0 or (ax_k / ay_k) > 3.0 or (ay_k / ax_k) > 3.0
        unraveled_to_fm = abs(f_k - f_fm) < 1e-4 or stats_k["is_uniform"]
        
        if diverged_grid:
            print(f"[{ansatz_name} Ansatz] Diverged scaling grid (ax={ax_k:.1f}, ay={ay_k:.1f}). Discarded.")
            continue
            
        if unraveled_to_fm:
            # We already have the true analytical FM energy populated, so ignore numeric approximations
            continue
            
        if actual_class == "Unknown Phase":
            print(f"[{ansatz_name} Ansatz] Relaxed into {stats_k['geometry']} with Q={stats_k['Q']:.2f}. Discarded as Unverified.")
            continue
            
        # Log defections gracefully
        if actual_class != ansatz_name:
            print(f"[{ansatz_name} Ansatz] defected and converged into -> {actual_class}! (Energy: {f_k:.5f})")
        else:
            print(f"[{ansatz_name} Ansatz] verified as pure {actual_class}. (Energy: {f_k:.5f})")
            
        # Add to the energy pool for the specific physical state it represents
        if actual_class not in final_energies or f_k < final_energies[actual_class]:
            final_energies[actual_class] = f_k
            best_states[actual_class] = (spins_k, ax_k, ay_k)
            
    # Determine the global numerical winner (the lowest valid energy branch)
    winner = min(final_energies, key=final_energies.get)
    print(f"\n=> The Ground State Phase is: {winner} (Energy: {final_energies[winner]:.5f})")
    
    # Extract the winning spins configuration for saving/plotting
    best_spins, best_ax, best_ay = None, 1.0, 1.0
    
    if winner == "FM":
        # Synthesize a pure continuous mathematical FM uniform state
        best_spins = np.zeros((L, L, 3))
        if f_fm == A_scaled - H_scaled:
            best_spins[:, :, 2] = 1.0 # Aligned UP
        elif f_fm == A_scaled + H_scaled:
            best_spins[:, :, 2] = -1.0 # Aligned DOWN
        else:
            nz = H_scaled / (2 * A_scaled)
            nx = np.sqrt(abs(1.0 - nz**2))
            best_spins[:, :, 0] = nx # Tilted
            best_spins[:, :, 2] = nz
    else:
        best_spins, best_ax, best_ay = best_states[winner]
            
    if best_spins is not None:
        out_name = f"output/LLG/Groundstates/LLG_groundstate_L{L}_A{A_scaled:.2f}_H{H_scaled:.2f}.npz"
        if save_outputs or plot_groundstate:
            np.savez(out_name, spins=best_spins, ax=best_ax, ay=best_ay)
            print(f"Saved numerical ground state to '{out_name}'")
        
        if plot_groundstate:
            try:
                from periodic_plotting import plot_periodic_structure
                print("Launching periodic plot...")
                plot_periodic_structure(out_name, tiles_x=2, tiles_y=2, display_mode="quiver", ax=best_ax, ay=best_ay)
            except Exception as e:
                print(f"Could not load periodic_plotting to display: {e}")
                
    return winner, final_energies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic LLG Phase Analyzer")
    parser.add_argument("--H", type=float, default=1.0, help="Scaled magnetic field")
    parser.add_argument("--A", type=float, default=0.8, help="Scaled Anisotropy")
    parser.add_argument("--L", type=int, default=32, help="Grid size L")
    parser.add_argument("--npy", type=str, default=None, help="Optional MC .npy or .npz file to use as an ansatz")
    parser.add_argument("--plot-ansatz", action="store_true", help="Plot each ansatz configuration before relaxing")
    parser.add_argument("--live-plot", action="store_true", help="Plot the real-time evolution of the solver")
    parser.add_argument("--live-mode", type=str, choices=["quiver", "heatmap"], default="quiver", help="Display mode for live plotting")
    parser.add_argument("--vis-scale", action="store_true", help="Turn on dynamic scaling visualization during live plotting")
    parser.add_argument("--plot-groundstate", action="store_true", help="Plot the final ground state configuration at the end")
    parser.add_argument("--plot-fft", action="store_true", help="Save Bragg peak FFT graphs of the relaxed phases to output directory")
    parser.add_argument("--max-dt", type=float, default=0.05, help="Maximum integration timestep")
    parser.add_argument("--cfl", type=float, default=0.25, help="CFL stability factor for dynamic timestep")
    
    args = parser.parse_args()
    
    compare_phases(H_scaled=args.H, A_scaled=args.A, L=args.L, npy_file=args.npy, plot_ansatz=args.plot_ansatz, live_plot=args.live_plot, live_mode=args.live_mode, max_dt=args.max_dt, cfl_factor=args.cfl, visualize_scaling=args.vis_scale, plot_groundstate=args.plot_groundstate, plot_fft=args.plot_fft)

