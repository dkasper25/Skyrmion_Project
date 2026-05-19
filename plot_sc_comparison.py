import numpy as np
import matplotlib.pyplot as plt
import types
from LLG_solver import init_SC, relax_phase
from fintemp_LLG import equilibrate_phase

def plot_pixelwise_comparison():
    print("Generating T=0 deterministic SC state (32x32)...")
    spins_init, ax_init, ay_init = init_SC(32)
    spins_t0, f_tot, ax_t0, ay_t0 = relax_phase(spins_init, 32, 1.0, 0.5, "SC", ax_in=ax_init, ay_in=ay_init, max_steps=50000, tol=1e-7, live_plot=False)
    print(f"T=0 State: ax={ax_t0:.4f}, ay={ay_t0:.4f}, Energy={f_tot:.5f}")

    print("\nGenerating Finite-Temperature SC state (32x32)...")
    # Mocking argparse for the equilibrate_phase function
    args = types.SimpleNamespace(
        H=1.0, A=0.5, T=1e-06, dt=0.005, steps=1000, block=50, 
        seed=42, no_plot=True, dynamic_scaling=True, live_mode="quiver"
    )
    
    # We pass the T=0 relaxed state as the starting point, just like the real pipeline does
    spins_ft, ax_ft, ay_ft, avg_energy, avg_terms = equilibrate_phase(
        spins_t0, 32, ax_t0, ay_t0, 1.0, 0.5, 1e-06, "SC", args
    )
    print(f"Finite-T State: ax={ax_ft:.4f}, ay={ay_ft:.4f}, Energy={avg_energy:.5f}")

    # Pixel-wise Comparison
    # 1. Angular difference (in degrees) between the 3D spin vectors
    dot_product = np.sum(spins_t0 * spins_ft, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_diff_deg = np.arccos(dot_product) * (180.0 / np.pi)

    # 2. Vector difference in the XY plane
    diff_u = spins_ft[:, :, 0] - spins_t0[:, :, 0]
    diff_v = spins_ft[:, :, 1] - spins_t0[:, :, 1]

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    
    L = 32
    
    # We will plot using pixel coordinates (0 to 31) since we are doing a pixel-wise comparison
    extent = [0, 32, 0, 32]
    X, Y = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5)

    # Subplot 1: T=0 Quiver
    U0 = spins_t0[:, :, 0].T
    V0 = spins_t0[:, :, 1].T
    Sz0 = spins_t0[:, :, 2].T
    q0 = ax0.quiver(X, Y, U0, V0, Sz0, cmap='bwr', pivot='mid', scale=L*1.2, width=0.005)
    ax0.set_title(f"T=0 Configuration (ax={ax_t0:.3f}, ay={ay_t0:.3f})")
    q0.set_clim(-1, 1)
    fig.colorbar(q0, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_xlim(0, 32)
    ax0.set_ylim(0, 32)
    ax0.set_aspect('equal')

    # Subplot 2: Finite T Quiver
    U1 = spins_ft[:, :, 0].T
    V1 = spins_ft[:, :, 1].T
    Sz1 = spins_ft[:, :, 2].T
    q1 = ax1.quiver(X, Y, U1, V1, Sz1, cmap='bwr', pivot='mid', scale=L*1.2, width=0.005)
    ax1.set_title(f"Finite-T Configuration (ax={ax_ft:.3f}, ay={ay_ft:.3f})")
    q1.set_clim(-1, 1)
    fig.colorbar(q1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xlim(0, 32)
    ax1.set_ylim(0, 32)
    ax1.set_aspect('equal')

    # Subplot 3: Vector Difference Field
    diff_z = spins_ft[:, :, 2] - spins_t0[:, :, 2]
    
    # Set background to dark to make the white (0 diff_z) vectors pop
    ax2.set_facecolor('#111111')
    
    # Scale reduced from 32.0 to 12.0 to make arrows significantly larger
    q2 = ax2.quiver(X, Y, diff_u.T, diff_v.T, diff_z.T, cmap='seismic', pivot='mid', scale=5.0, width=0.003, headwidth=4)
    ax2.set_title(r"Vector Difference ($\vec{S}_{FT} - \vec{S}_{T=0}$): $\Delta S_z$ (Color) & In-Plane Shift (Arrows)")
    
    # max possible difference is 2. Let's scale clim slightly to make colors pop
    max_dz = np.max(np.abs(diff_z))
    if max_dz < 0.1: max_dz = 0.1
    q2.set_clim(-max_dz, max_dz)
    
    fig.colorbar(q2, ax=ax2, fraction=0.02, pad=0.04, label="$\Delta S_z$")
    ax2.set_xlim(0, 32)
    ax2.set_ylim(0, 32)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("output/SC_pixelwise_diff.png", dpi=300)
    print("Saved pixel-wise comparison plot to output/SC_pixelwise_diff.png")

if __name__ == "__main__":
    plot_pixelwise_comparison()
