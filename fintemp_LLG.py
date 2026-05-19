import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import diffrax
import equinox as eqx
import lineax as lx
import argparse
import os
from LLG_solver import init_SkX, init_SC, init_SP, relax_phase, get_FM_energy, analyze_state, validate_result
# ---------------------------------------------------------
# SDE LLG using JAX + Diffrax
# ---------------------------------------------------------

@jax.jit
def get_effective_field(spins, ax, ay, H_scaled, A_scaled):
    """Compute the deterministic effective field using jax.numpy."""
    # Periodic boundaries
    n_right = jnp.roll(spins, shift=-1, axis=0)
    n_left  = jnp.roll(spins, shift=1, axis=0)
    n_up    = jnp.roll(spins, shift=-1, axis=1) # -1 on axis 1 means shifting elements to lower index (up)
    n_down  = jnp.roll(spins, shift=1, axis=1)
    
    n_x = spins[..., 0]
    n_y = spins[..., 1]
    n_z = spins[..., 2]
    
    # Exchange field
    H_x = (n_right[..., 0] + n_left[..., 0] - 2*n_x)/(ax**2) + (n_up[..., 0] + n_down[..., 0] - 2*n_x)/(ay**2)
    H_y = (n_right[..., 1] + n_left[..., 1] - 2*n_y)/(ax**2) + (n_up[..., 1] + n_down[..., 1] - 2*n_y)/(ay**2)
    H_z = (n_right[..., 2] + n_left[..., 2] - 2*n_z)/(ax**2) + (n_up[..., 2] + n_down[..., 2] - 2*n_z)/(ay**2)
    
    # DMI field (Chirality strictly maintained from deterministic solver)
    H_x += (n_right[..., 2] - n_left[..., 2]) / ax
    H_y += (n_up[..., 2] - n_down[..., 2]) / ay
    H_z += - (n_right[..., 0] - n_left[..., 0]) / ax - (n_up[..., 1] - n_down[..., 1]) / ay
    
    # Zeeman & Anisotropy
    H_z -= H_scaled
    H_z -= 2 * A_scaled * n_z
    
    return jnp.stack([H_x, H_y, H_z], axis=-1)

@jax.jit
def get_energy_density(y, ax, ay, H_scaled, A_scaled):
    """Calculate the spatial average of the deterministic energy density evaluated strictly on normalized states."""
    norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    y_unit = y / jnp.where(norm > 1e-12, norm, 1.0)
    
    y_right = jnp.roll(y_unit, shift=-1, axis=0)
    y_up = jnp.roll(y_unit, shift=-1, axis=1)
    
    E_ex_x = 0.5 * jnp.sum((y_right - y_unit)**2, axis=-1)
    E_ex_y = 0.5 * jnp.sum((y_up - y_unit)**2, axis=-1)
    
    E_dmi_x = y_unit[..., 2] * (y_right[..., 0] - y_unit[..., 0]) - y_unit[..., 0] * (y_right[..., 2] - y_unit[..., 2])
    E_dmi_y = y_unit[..., 2] * (y_up[..., 1] - y_unit[..., 1]) - y_unit[..., 1] * (y_up[..., 2] - y_unit[..., 2])
    
    E_z = H_scaled * y_unit[..., 2]
    E_a = A_scaled * y_unit[..., 2]**2
    
    f_ex = jnp.mean((E_ex_x / ax**2) + (E_ex_y / ay**2))
    f_dmi = jnp.mean((E_dmi_x / ax) + (E_dmi_y / ay))
    f_z = jnp.mean(E_z)
    f_a = jnp.mean(E_a)
    f_tot = f_ex + f_dmi + f_z + f_a
    
    # Pure dimensionless per-bond energies
    norm_ex_x = jnp.mean(E_ex_x)
    norm_ex_y = jnp.mean(E_ex_y)
    norm_dmi_x = jnp.mean(E_dmi_x)
    norm_dmi_y = jnp.mean(E_dmi_y)
    
    return f_tot, f_ex, f_dmi, f_z, f_a, norm_ex_x, norm_ex_y, norm_dmi_x, norm_dmi_y, f_z, f_a

def drift_fn(t, y, args):
    """Deterministic LLG Step: dn/dt = H_eff - (n . H_eff)n"""
    ax, ay, H_scaled, A_scaled, sigma = args
    # Explicit SDE solvers evaluate intermediate stages that drift off the unit sphere |n|=1.
    # LLG has a cubic non-linearity (n . H)n. If |n|>1, this explodes rapidly.
    # We enforce evaluation strictly on the normalized manifold.
    norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    y_unit = y / jnp.where(norm > 1e-12, norm, 1.0)
    
    H_eff = get_effective_field(y_unit, ax, ay, H_scaled, A_scaled)
    dot_val = jnp.sum(y_unit * H_eff, axis=-1, keepdims=True)
    return H_eff - dot_val * y_unit

def diffusion_fn(t, y, args):
    """Linear operator for Stratonovich geometric noise projection."""
    ax, ay, H_scaled, A_scaled, sigma = args
    norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    y_unit = y / jnp.where(norm > 1e-12, norm, 1.0)
    
    def custom_mv(dW):
        dot = jnp.sum(y_unit * dW, axis=-1, keepdims=True)
        return sigma * (dW - dot * y_unit)
    return lx.FunctionLinearOperator(custom_mv, input_structure=jax.eval_shape(lambda: y))

@eqx.filter_jit
def simulate_single_block(t0, y0, args, key, sigma, dt, block_steps):
    """Run full stochastic Heun integration for a single block interval."""
    t1 = t0 + block_steps * dt
    # VirtualBrownianTree generates the stochastic paths deterministically seeded by 'key'
    bm = diffrax.VirtualBrownianTree(t0=t0, t1=t1, tol=dt/2, shape=y0.shape, key=key)
    
    drift = diffrax.ODETerm(drift_fn)
    diffusion = diffrax.ControlTerm(diffusion_fn, bm)
    # Both terms form the SDE
    terms = diffrax.MultiTerm(drift, diffusion)
    
    # EulerHeun handles Stratonovich SDEs correctly
    solver = diffrax.EulerHeun()
    
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0=dt,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=block_steps+10 # Allow buffer
    )
    
    y_final = sol.ys[-1]
    
    # Strict geometrical projection to |n|=1 to kill any numerical random walk drift
    norm = jnp.linalg.norm(y_final, axis=-1, keepdims=True)
    y_norm_final = y_final / jnp.where(norm > 1e-12, norm, 1.0)
    
    return t1, y_norm_final

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def load_ansatz(filepath):
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
        spins = np.load(filepath)
        
    return spins, ax, ay

def init_FM(L, H_scaled, A_scaled):
    spins = np.zeros((L, L, 3))
    e_aligned = A_scaled + H_scaled 
    e_anti_aligned = A_scaled - H_scaled 
    e_tilted = float('inf')
    nz_tilted = 1.0
    if A_scaled != 0:
        nz_tilted = H_scaled / (2 * A_scaled)
        if abs(nz_tilted) <= 1.0:
            e_tilted = - (H_scaled**2) / (4 * A_scaled)
            
    min_e = min(e_aligned, e_anti_aligned, e_tilted)
    if min_e == e_aligned:
        spins[:, :, 2] = 1.0
    elif min_e == e_anti_aligned:
        spins[:, :, 2] = -1.0
    else:
        spins[:, :, 2] = nz_tilted
        spins[:, :, 0] = np.sqrt(1.0 - nz_tilted**2)
        
    return spins, 1.0, 1.0

# ---------------------------------------------------------
# Phase Evaluation & Equilibration
# ---------------------------------------------------------

def equilibrate_phase(spins_np, L_run, ax, ay, H_scaled, A_scaled, T_scaled, phase_name, args):
    """Run the SDE thermalization block and return the thermal average energy."""
    L_x, L_y, _ = spins_np.shape
    gamma = ay / ax

    
    if L_run is not None and L_run > L_x:
        reps = L_run // L_x
        spins_np = np.tile(spins_np, (reps, reps, 1))
        L_x, L_y = spins_np.shape[0], spins_np.shape[1]
    
    # Thermodynamic proper SDE noise scaling for continuous finite-volume cells
    sigma = np.sqrt(2.0 * T_scaled / (ax * ay))
    y0 = jnp.array(spins_np, dtype=jnp.float64)
    t0 = jnp.array(0.0, dtype=jnp.float64)
    sim_args = (jnp.float64(ax), jnp.float64(ay), jnp.float64(H_scaled), jnp.float64(A_scaled), jnp.float64(sigma))
    
    live_plot = not args.no_plot
    live_mode = args.live_mode
    if live_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax_plot = plt.subplots(figsize=(8,8))
        if live_mode == "quiver":
            tiles_x, tiles_y = 1, 1 
            tiled_spins = np.tile(spins_np, (tiles_x, tiles_y, 1))
            L_x_tiled, L_y_tiled = tiled_spins.shape[0], tiled_spins.shape[1]
            X_base, Y_base = np.meshgrid(np.arange(L_x_tiled), np.arange(L_y_tiled))
            X, Y = X_base * ax, Y_base * ay
            U = tiled_spins[:, :, 0].T
            V = tiled_spins[:, :, 1].T
            Sz = tiled_spins[:, :, 2].T
            q = ax_plot.quiver(X, Y, U, V, Sz, cmap="coolwarm", pivot='mid', scale=max(L_x_tiled, L_y_tiled)*0.8, width=0.005)
            q.set_clim(-1, 1)
            rects = []
            for i in range(tiles_x):
                for j in range(tiles_y):
                    rect = plt.Rectangle(((-0.5 + i*L_x)*ax, (-0.5 + j*L_y)*ay), L_x*ax, L_y*ay, 
                                       fill=False, edgecolor='black', linestyle='--', linewidth=1.5, alpha=0.5)
                    ax_plot.add_patch(rect)
                    rects.append((i, j, rect))
            ax_plot.set_xlim(-ax, L_x_tiled * ax)
            ax_plot.set_ylim(-ay, L_y_tiled * ay)
            ax_plot.set_aspect('equal')
        else:
            im = ax_plot.imshow(spins_np[:, :, 2].T, vmin=-1, vmax=1, cmap='bwr', origin='lower', extent=[0, L_x*ax, 0, L_y*ay])
            ax_plot.set_aspect('equal')
        plt.show()

    key = jax.random.PRNGKey(args.seed)
    energy_history = []
    energy_terms_history = {'ex': [], 'dmi': [], 'z': [], 'a': []}
    
    key, subkey = jax.random.split(key)
    print(f"[{phase_name}] Simulating {args.steps} SDE steps (block iterative)...")
    
    num_blocks = args.steps // args.block
    for i in range(num_blocks):
        key, subkey = jax.random.split(key)
        t_next, y_next = simulate_single_block(t0, y0, sim_args, subkey, sigma, args.dt, args.block)
        t0 = t_next
        y0 = y_next
        
        # Evaluate thermodynamic energy for the current snapshot
        vals = get_energy_density(y0, jnp.float64(ax), jnp.float64(ay), jnp.float64(H_scaled), jnp.float64(A_scaled))
        f_val, f_ex, f_dmi, f_z, f_a, norm_ex_x, norm_ex_y, norm_dmi_x, norm_dmi_y, norm_z, norm_a = [float(x) for x in vals]
        
        # Adiabatic Barostat Update
        if getattr(args, 'dynamic_scaling', False):
            alpha_scale = 0.01 if T_scaled == 0 else 0.05
            noise_threshold = 1e-12 if T_scaled == 0 else 1e-4
            if getattr(args, 'iso_scale', False):
                dmi_term = abs(norm_dmi_x + norm_dmi_y / gamma)
                if dmi_term > noise_threshold and (norm_ex_x + norm_ex_y) > noise_threshold:
                    target_ax = 2.0 * (norm_ex_x + norm_ex_y / gamma**2) / dmi_term
                    target_ay = target_ax * gamma
                    ax = (1.0 - alpha_scale) * ax + alpha_scale * target_ax
                    ay = (1.0 - alpha_scale) * ay + alpha_scale * target_ay
            else:
                if abs(norm_dmi_x) > noise_threshold and abs(norm_ex_x) > noise_threshold:
                    target_ax = 2.0 * norm_ex_x / abs(norm_dmi_x)
                    ax = (1.0 - alpha_scale) * ax + alpha_scale * target_ax
                if abs(norm_dmi_y) > noise_threshold and abs(norm_ex_y) > noise_threshold:
                    target_ay = 2.0 * norm_ex_y / abs(norm_dmi_y)
                    ay = (1.0 - alpha_scale) * ay + alpha_scale * target_ay
                
                # Enforce isotropy for 1D states (prevent severe aspect ratio warping)
                if abs(norm_dmi_y) <= noise_threshold and abs(norm_ex_y) <= noise_threshold and abs(norm_dmi_x) > noise_threshold:
                    ay = ax
                if abs(norm_dmi_x) <= noise_threshold and abs(norm_ex_x) <= noise_threshold and abs(norm_dmi_y) > noise_threshold:
                    ax = ay
                
            sigma = np.sqrt(2.0 * T_scaled / (ax * ay))
            sim_args = (jnp.float64(ax), jnp.float64(ay), jnp.float64(H_scaled), jnp.float64(A_scaled), jnp.float64(sigma))
        
        steps_done = (i + 1) * args.block
        norm_ex = norm_ex_x + norm_ex_y
        norm_dmi = norm_dmi_x + norm_dmi_y
        
        # Wait until 30% of simulation has passed to clear initial transients
        if steps_done > 0.3 * args.steps:
            energy_history.append(f_val)
            energy_terms_history['ex'].append(norm_ex)
            energy_terms_history['dmi'].append(norm_dmi)
            energy_terms_history['z'].append(norm_z)
            energy_terms_history['a'].append(norm_a)
            
        if live_plot and (args.block >= 10 or (steps_done // 10 > (steps_done - args.block) // 10)):
            current_spins_np = np.asarray(y0)
            if live_mode == "quiver":
                tiled_spins = np.tile(current_spins_np, (tiles_x, tiles_y, 1))
                U = tiled_spins[:, :, 0].T
                V = tiled_spins[:, :, 1].T
                Sz = tiled_spins[:, :, 2].T
                q.set_UVC(U, V, Sz)
                
                # Update positions for physical scaling
                pts = np.column_stack((X_base.flatten() * ax, Y_base.flatten() * ay))
                q.set_offsets(pts)
                
                for (rect_i, rect_j, rect) in rects:
                    rect.set_xy(((-0.5 + rect_i*L_x)*ax, (-0.5 + rect_j*L_y)*ay))
                    rect.set_width(L_x * ax)
                    rect.set_height(L_y * ay)
                    
                ax_plot.set_xlim(-ax, L_x_tiled * ax)
                ax_plot.set_ylim(-ay, L_y_tiled * ay)
            else:
                im.set_data(current_spins_np[:, :, 2].T)
                im.set_extent([0, L_x*ax, 0, L_y*ay])
                ax_plot.set_xlim(0, L_x*ax)
                ax_plot.set_ylim(0, L_y*ay)
                
            ax_plot.set_title(f"[{phase_name}] T={T_scaled} | Steps: {steps_done}/{args.steps} | f_inst: {f_val:.4f} | ax: {ax:.3f}")
            plt.pause(0.01)
        else:
            if steps_done % (args.block * 50) == 0:
                print(f"[{phase_name}] Progress: {steps_done}/{args.steps} steps... f_inst={f_val:.4f} (ax={ax:.3f}, ay={ay:.3f})")
        
    if live_plot:
        plt.ioff()
        plt.close(fig)
        
    avg_energy = np.mean(energy_history) if energy_history else f_val
    avg_terms = {
        'ex': np.mean(energy_terms_history['ex']) if energy_terms_history['ex'] else f_ex,
        'dmi': np.mean(energy_terms_history['dmi']) if energy_terms_history['dmi'] else f_dmi,
        'z': np.mean(energy_terms_history['z']) if energy_terms_history['z'] else f_z,
        'a': np.mean(energy_terms_history['a']) if energy_terms_history['a'] else f_a
    }
    print(f"[{phase_name}] Equilibration complete. Thermal Energy Density: {avg_energy:.5f}")
    return np.asarray(y0), ax, ay, avg_energy, avg_terms

def compare_fintemp_phases(args, save_outputs=True):
    H_scaled = args.H
    A_scaled = args.A
    T_scaled = args.T
    L_ansatz = args.L
    L_super = args.L_super
    
    print(f"--- Finite-Temp Phase Evaluation H={H_scaled}, As={A_scaled}, T={T_scaled} ---")
    print(f"JAX Backend: {jax.default_backend()}")
    results = {}
    results_terms = {}
    states = {}
    
    phases = [
        ("SkX", init_SkX),
        ("SC", init_SC),
        ("SP", init_SP),
        ("FM", lambda L: init_FM(L, H_scaled, A_scaled))
    ]
    
    f_fm_analytical = get_FM_energy(H_scaled, A_scaled)
    
    for phase_name, init_fn in phases:
        if phase_name == "FM":
            print(f"\nEvaluating {phase_name}...")
            spins, ax, ay = init_fn(L_ansatz)
            final_spins, f_ax, f_ay, avg_energy, avg_terms = equilibrate_phase(spins, L_super, ax, ay, H_scaled, A_scaled, T_scaled, phase_name, args)
            
            stats = analyze_state(final_spins, f_ax, f_ay, phase_name=f"{phase_name}_fintemp", plot_fft=getattr(args, 'plot_fft', False))
            actual_class = stats["classified_state"]
            print(f"   -> Q={stats['Q']:.2f} | Geo: {stats['geometry']} ({stats['peaks']} Peaks) -> Detected as: {actual_class}")
            
            if actual_class != phase_name:
                print(f"[{phase_name} Ansatz] defected and converged into -> {actual_class}! (Thermal Energy: {avg_energy:.5f})")
            else:
                print(f"[{phase_name} Ansatz] verified as pure {actual_class}. (Thermal Energy: {avg_energy:.5f})")

            if actual_class not in results or avg_energy < results[actual_class]:
                results[actual_class] = avg_energy
                results_terms[actual_class] = avg_terms
                states[actual_class] = (final_spins, f_ax, f_ay)
        else:
            print(f"\nRelaxing {phase_name} Phase at T=0 to optimize boundaries...")
            spins_init, ax_init, ay_init = init_fn(L_ansatz)
            spins, f_tot_0K, ax, ay = relax_phase(spins_init, L_ansatz, H_scaled, A_scaled, phase_name, ax_in=ax_init, ay_in=ay_init, max_steps=100000, tol=1e-12, live_plot=False, iso_scale=getattr(args, 'iso_scale', False))
            
            # Use raw nz variance — consistent with analyze_state's is_uniform definition.
            # Aspect ratio threshold matches LLG_solver.py (> 3).
            spin_variance = np.var(spins[:, :, 2])
            diverged = ax > 50.0 or ay > 50.0 or (ax / ay) > 3.0 or (ay / ax) > 3.0
            
            if abs(f_tot_0K - f_fm_analytical) < 1e-4 or spin_variance < 1e-4 or diverged:
                print(f"[{phase_name}] unraveled directly into a uniform state (Periodicity Diverged). Skipping redundant finite-T SDE mapping!")
                continue
                
            print(f"Beginning Finite-Temperature SDE equilibration...")
            final_spins, f_ax, f_ay, avg_energy, avg_terms = equilibrate_phase(spins, L_super, ax, ay, H_scaled, A_scaled, T_scaled, phase_name, args)
            
            stats = analyze_state(final_spins, f_ax, f_ay, phase_name=f"{phase_name}_fintemp", plot_fft=getattr(args, 'plot_fft', False))
            actual_class = stats["classified_state"]
            print(f"   -> Q={stats['Q']:.2f} | Geo: {stats['geometry']} ({stats['peaks']} Peaks) -> Detected as: {actual_class}")
            
            # Post-SDE validation: same gate as compare_phases uses after T=0 relaxation.
            # Catches grid drift or FM collapse that occurred *during* thermalization.
            discard_reason = validate_result(phase_name, avg_energy, f_ax, f_ay, f_fm_analytical, stats)
            if discard_reason:
                print(f"[{phase_name} Ansatz] Post-SDE: {discard_reason}")
                continue
            
            if actual_class != phase_name:
                print(f"[{phase_name} Ansatz] defected and converged into -> {actual_class}! (Thermal Energy: {avg_energy:.5f})")
            else:
                print(f"[{phase_name} Ansatz] verified as pure {actual_class}. (Thermal Energy: {avg_energy:.5f})")

            if actual_class not in results or avg_energy < results[actual_class]:
                results[actual_class] = avg_energy
                results_terms[actual_class] = avg_terms
                states[actual_class] = (final_spins, f_ax, f_ay)
        
    # Analyze the winner
    winner = min(results, key=results.get)
    print(f"\n=> The Ground State Phase at T={T_scaled} is: {winner} (Average Thermal Energy: {results[winner]:.5f})")
    
    # Save the winner
    if save_outputs:
        best_spins, best_ax, best_ay = states[winner]
        out_dir = "output/Fintemp"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"fintemp_groundstate_T{T_scaled}_H{H_scaled}.npz")
        np.savez(out_file, spins=best_spins, ax=best_ax, ay=best_ay)
        print(f"Saved finite-temperature ground state to '{out_file}'")

    if getattr(args, 'save_all', False):
        os.makedirs("output/Fintemp", exist_ok=True)
        for phase, (s_spins, s_ax, s_ay) in states.items():
            L_x = s_spins.shape[0]
            # Extract a single unit cell if the state was tiled
            if L_x > L_ansatz and L_x % L_ansatz == 0:
                s_spins_cell = s_spins[:L_ansatz, :L_ansatz, :]
                out_file_cell = os.path.join("output/Fintemp", f"fintemp_relaxed_{phase}_cell_T{T_scaled}_H{H_scaled}.npz")
                np.savez(out_file_cell, spins=s_spins_cell, ax=s_ax, ay=s_ay)
                print(f"Saved finite-temperature {phase} state (single cell) to '{out_file_cell}'")
                
            out_file_super = os.path.join("output/Fintemp", f"fintemp_relaxed_{phase}_super_T{T_scaled}_H{H_scaled}.npz")
            np.savez(out_file_super, spins=s_spins, ax=s_ax, ay=s_ay)
            print(f"Saved finite-temperature {phase} state (full supercell) to '{out_file_super}'")
    
    return winner, results, results_terms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-Temperature SDE LLG Phase Evaluator")
    parser.add_argument("--L", type=int, default=32, help="Grid size L for ansatz creation")
    parser.add_argument("--L_super", type=int, default=64, help="Target supercell size (tiles the initial lattice)")
    parser.add_argument("--H", type=float, default=1, help="Scaled magnetic field")
    parser.add_argument("--A", type=float, default=0.5, help="Scaled Anisotropy")
    parser.add_argument("--T", type=float, default=0.05, help="Dimensionless Temperature")
    parser.add_argument("--dt", type=float, default=0.005, help="Integration step size")
    parser.add_argument("--steps", type=int, default=5000, help="Total SDE thermalization steps")
    parser.add_argument("--block", type=int, default=50, help="Steps per block (for normalization & visualization)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PRNG")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plotting")
    parser.add_argument("--live-mode", type=str, choices=["quiver", "heatmap"], default="quiver", help="Display mode for live plotting")
    parser.add_argument("--plot-fft", action="store_true", help="Save Bragg peak FFT graphs of the relaxed phases to output directory")
    parser.add_argument("--save-all", action="store_true", help="Save all relaxed phases (extracts single cell if tiled)")
    parser.add_argument("--dynamic-scaling", action="store_true", help="Enable adiabatic barostat to dynamically rescale ax and ay during equilibration")
    parser.add_argument("--iso-scale", action="store_true", help="Enforce isotropic scaling (preserve initial aspect ratio)")
    
    args = parser.parse_args()
    compare_fintemp_phases(args)
