import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import lineax as lx
import argparse
import os
from LLG_solver import init_SkX, init_SC, init_SP, relax_phase, get_FM_energy

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
    
    E_tot = (E_ex_x / ax**2) + (E_ex_y / ay**2) + (E_dmi_x / ax) + (E_dmi_y / ay) + E_z + E_a
    f_tot = jnp.mean(E_tot)
    return f_tot

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
def simulate_block(t0, y0, args, key, sigma, dt, num_steps):
    """Run a single block (chunk) of stochastic Heun integration."""
    t1 = t0 + num_steps * dt
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
        max_steps=num_steps+2 # Allow buffer
    )
    
    y_final = sol.ys[0] 
    
    # Strict geometrical projection to |n|=1 to kill any numerical random walk drift
    norm = jnp.linalg.norm(y_final, axis=-1, keepdims=True)
    y_norm = y_final / jnp.where(norm > 1e-12, norm, 1.0)
    
    return t1, y_norm

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
    
    if L_run is not None and L_run > L_x:
        reps = L_run // L_x
        spins_np = np.tile(spins_np, (reps, reps, 1))
        L_x, L_y = spins_np.shape[0], spins_np.shape[1]
    
    sigma = np.sqrt(2.0 * T_scaled)
    y0 = jnp.array(spins_np, dtype=jnp.float32)
    t0 = jnp.array(0.0)
    sim_args = (jnp.float32(ax), jnp.float32(ay), jnp.float32(H_scaled), jnp.float32(A_scaled), jnp.float32(sigma))
    
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
    steps_done = 0
    energy_history = []
    
    while steps_done < args.steps:
        key, subkey = jax.random.split(key)
        current_block_steps = min(args.block, args.steps - steps_done)
        
        t0, y0 = simulate_block(t0, y0, sim_args, subkey, sigma, args.dt, current_block_steps)
        steps_done += current_block_steps
        
        # Evaluate thermodynamic energy at block boundaries
        f_val = float(get_energy_density(y0, ax, ay, H_scaled, A_scaled))
        # Wait until 30% of simulation has passed to clear initial transients
        if steps_done > 0.3 * args.steps:
            energy_history.append(f_val)
            
        if live_plot:
            current_spins_np = np.asarray(y0)
            if live_mode == "quiver":
                tiled_spins = np.tile(current_spins_np, (tiles_x, tiles_y, 1))
                U = tiled_spins[:, :, 0].T
                V = tiled_spins[:, :, 1].T
                Sz = tiled_spins[:, :, 2].T
                q.set_UVC(U, V, Sz)
            else:
                im.set_data(current_spins_np[:, :, 2].T)
            ax_plot.set_title(f"[{phase_name}] T={T_scaled} | Steps: {steps_done}/{args.steps} | f_inst: {f_val:.4f}")
            plt.pause(0.01)
        else:
            if steps_done % (args.block * 10) == 0:
                print(f"[{phase_name}] Progress: {steps_done}/{args.steps} steps... f_inst={f_val:.4f}")
                
    if live_plot:
        plt.ioff()
        plt.close(fig)
        
    avg_energy = np.mean(energy_history) if energy_history else f_val
    print(f"[{phase_name}] Equilibration complete. Thermal Energy Density: {avg_energy:.5f}")
    return np.asarray(y0), ax, ay, avg_energy

def compare_fintemp_phases(args, save_outputs=True):
    H_scaled = args.H
    A_scaled = args.A
    T_scaled = args.T
    L_ansatz = args.L
    L_super = args.L_super
    
    print(f"--- Finite-Temp Phase Evaluation H={H_scaled}, As={A_scaled}, T={T_scaled} ---")
    print(f"JAX Backend: {jax.default_backend()}")
    results = {}
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
            final_spins, f_ax, f_ay, avg_energy = equilibrate_phase(spins, L_super, ax, ay, H_scaled, A_scaled, T_scaled, phase_name, args)
            results[phase_name] = avg_energy
            states[phase_name] = (final_spins, f_ax, f_ay)
        else:
            print(f"\nRelaxing {phase_name} Phase at T=0 to optimize boundaries...")
            spins_init, ax_init, ay_init = init_fn(L_ansatz)
            spins, f_tot_0K, ax, ay = relax_phase(spins_init, L_ansatz, H_scaled, A_scaled, phase_name, ax_in=ax_init, ay_in=ay_init, max_steps=50000, tol=1e-7, live_plot=False)
            print(f"-> T=0 Relaxation complete. Optimized bounds: ax={ax:.4f}, ay={ay:.4f} (Deterministic Energy: {f_tot_0K:.5f})")
            # Additional structural check: if variance across lattice approaches zero, it is physically FM
            # If the boundary dimensions geometrically diverge (e.g. ax > 5.0 meaning the period stretches 
            # longer than the physical simulation scope), the structure is melting into a uniform phase.
            spin_variance = np.var(spins, axis=(0, 1)).sum()
            diverged = ax > 50.0 or ay > 50.0 or (ax / ay) > 100.0 or (ay / ax) > 100.0
            
            if abs(f_tot_0K - f_fm_analytical) < 1e-4 or spin_variance < 1e-4 or diverged:
                print(f"[{phase_name}] unraveled directly into a uniform state (Periodicity Diverged). Skipping redundant finite-T SDE mapping!")
                continue
                
            print(f"Beginning Finite-Temperature SDE equilibration...")
            final_spins, f_ax, f_ay, avg_energy = equilibrate_phase(spins, L_super, ax, ay, H_scaled, A_scaled, T_scaled, phase_name, args)
            results[phase_name] = avg_energy
            states[phase_name] = (final_spins, f_ax, f_ay)
        
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
    
    return winner, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-Temperature SDE LLG Phase Evaluator")
    parser.add_argument("--L", type=int, default=64, help="Grid size L for ansatz creation")
    parser.add_argument("--L_super", type=int, default=None, help="Target supercell size (tiles the initial lattice)")
    parser.add_argument("--H", type=float, default=0.08, help="Scaled magnetic field")
    parser.add_argument("--A", type=float, default=0.5, help="Scaled Anisotropy")
    parser.add_argument("--T", type=float, default=0.05, help="Dimensionless Temperature")
    parser.add_argument("--dt", type=float, default=0.005, help="Integration step size")
    parser.add_argument("--steps", type=int, default=5000, help="Total SDE thermalization steps")
    parser.add_argument("--block", type=int, default=50, help="Steps per block (for normalization & visualization)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PRNG")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plotting")
    parser.add_argument("--live-mode", type=str, choices=["quiver", "heatmap"], default="quiver", help="Display mode for live plotting")
    
    args = parser.parse_args()
    compare_fintemp_phases(args)
