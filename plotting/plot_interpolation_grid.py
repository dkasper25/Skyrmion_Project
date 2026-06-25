import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

# Allow imports from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLG_solver import init_SkX, init_SC, relax_phase
from fintemp_LLG import interpolate_spins_periodic

def main():
    print("=== Generating Data for Grid Interpolation Plot ===")
    L = 8
    H = 0.08
    A = 0.5
    a_com = 0.50
    
    # 1. Initialize and relax SkX phase
    print("\n[1/4] Initializing and relaxing SkX phase...")
    spins_skx_init, ax_skx_init, ay_skx_init = init_SkX(L)
    spins_skx_rel, f_skx, ax_skx_rel, ay_skx_rel = relax_phase(
        spins_skx_init, L, H, A, "SkX", 
        ax_in=ax_skx_init, ay_in=ay_skx_init, 
        max_steps=20000, tol=1e-10, live_plot=False, iso_scale=True
    )
    
    # 2. Interpolate SkX phase
    print("\n[2/4] Interpolating SkX phase...")
    P_x_skx, P_y_skx = L * ax_skx_rel, L * ay_skx_rel
    Lx_skx_new = max(1, int(round(P_x_skx / a_com)))
    Ly_skx_new = max(1, int(round(P_y_skx / a_com)))
    ax_skx_new = a_com
    ay_skx_new = a_com
    spins_skx_int = interpolate_spins_periodic(spins_skx_rel, Lx_skx_new, Ly_skx_new)
    
    print(f"SkX Relaxed: {L}x{L} @ ax={ax_skx_rel:.4f}, ay={ay_skx_rel:.4f} (Box: {P_x_skx:.3f} x {P_y_skx:.3f})")
    print(f"SkX Interpolated: {Lx_skx_new}x{Ly_skx_new} @ ax={ax_skx_new:.4f}, ay={ay_skx_new:.4f}")
    
    # 3. Initialize and relax SC phase
    print("\n[3/4] Initializing and relaxing SC phase...")
    spins_sc_init, ax_sc_init, ay_sc_init = init_SC(L)
    spins_sc_rel, f_sc, ax_sc_rel, ay_sc_rel = relax_phase(
        spins_sc_init, L, H, A, "SC", 
        ax_in=ax_sc_init, ay_in=ay_sc_init, 
        max_steps=20000, tol=1e-10, live_plot=False, iso_scale=True
    )
    
    # 4. Interpolate SC phase
    print("\n[4/4] Interpolating SC phase...")
    P_x_sc, P_y_sc = L * ax_sc_rel, L * ay_sc_rel
    Lx_sc_new = max(1, int(round(P_x_sc / a_com)))
    Ly_sc_new = max(1, int(round(P_y_sc / a_com)))
    ax_sc_new = a_com
    ay_sc_new = a_com
    spins_sc_int = interpolate_spins_periodic(spins_sc_rel, Lx_sc_new, Ly_sc_new)
    
    print(f"SC Relaxed: {L}x{L} @ ax={ax_sc_rel:.4f}, ay={ay_sc_rel:.4f} (Box: {P_x_sc:.3f} x {P_y_sc:.3f})")
    print(f"SC Interpolated: {Lx_sc_new}x{Ly_sc_new} @ ax={ax_sc_new:.4f}, ay={ay_sc_new:.4f}")
    
    # -------------------------------------------------------------
    # Plotting code
    # -------------------------------------------------------------
    print("\nGenerating high-quality plot...")
    
    # Set up publication-style plot aesthetics
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
    
    # helper function to plot a single panel
    def plot_panel(ax_plot, spins, L_x, L_y, ax_val, ay_val, title, dot_size=18, show_ylabel=True):
        # Calculate coordinate meshes
        X, Y = np.meshgrid(np.arange(L_x) * ax_val, np.arange(L_y) * ay_val, indexing='ij')
        
        # Extent to align pixel centers perfectly with grid points
        extent = [-0.5 * ax_val, (L_x - 0.5) * ax_val, -0.5 * ay_val, (L_y - 0.5) * ay_val]
        
        # Plot soft background heatmap of Sz
        im = ax_plot.imshow(
            spins[:, :, 2].T, 
            cmap='bwr', 
            vmin=-1, 
            vmax=1, 
            origin='lower', 
            extent=extent, 
            alpha=0.6,
            interpolation='bicubic'
        )
        
        # Plot grid points as black dots with a fine white border for maximum contrast
        ax_plot.scatter(
            X.flatten(), 
            Y.flatten(), 
            color='black', 
            edgecolors='white', 
            linewidths=0.4, 
            s=dot_size, 
            zorder=3, 
            label='Grid Probe Points'
        )
        
        # Draw physical boundary box of the unit cell
        rect = plt.Rectangle(
            (-0.5 * ax_val, -0.5 * ay_val), 
            L_x * ax_val, 
            L_y * ay_val, 
            fill=False, 
            edgecolor='#333333', 
            linestyle='--', 
            linewidth=1.2, 
            alpha=0.8,
            zorder=2
        )
        ax_plot.add_patch(rect)
        
        ax_plot.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax_plot.set_xlabel('Physical X', fontsize=9, labelpad=4)
        if show_ylabel:
            ax_plot.set_ylabel('Physical Y', fontsize=9, labelpad=4)
        
        # Set aspect ratio to equal to avoid distortion of physical dimensions
        ax_plot.set_aspect('equal')
        
        # Set tight limits around the physical unit cell
        ax_plot.set_xlim(-1.2 * ax_val, (L_x + 0.2) * ax_val)
        ax_plot.set_ylim(-1.2 * ay_val, (L_y + 0.2) * ay_val)
        
        # Show clean grid lines corresponding to the physical coordinates
        ax_plot.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        
        return im

    # Row 1: SkX Phase
    im1 = plot_panel(
        axs[0, 0], spins_skx_rel, L, L, ax_skx_rel, ay_skx_rel, 
        f"SkX Relaxed (T=0)\nGrid: {L}x{L} | $a_x$={ax_skx_rel:.3f}, $a_y$={ay_skx_rel:.3f}",
        dot_size=30
    )
    im2 = plot_panel(
        axs[0, 1], spins_skx_int, Lx_skx_new, Ly_skx_new, ax_skx_new, ay_skx_new, 
        f"SkX Interpolated (Standard $a$)\nGrid: {Lx_skx_new}x{Ly_skx_new} | $a_x$=$a_y$={a_com:.3f}",
        dot_size=10,
        show_ylabel=False
    )
    
    # Row 2: SC Phase
    im3 = plot_panel(
        axs[1, 0], spins_sc_rel, L, L, ax_sc_rel, ay_sc_rel, 
        f"SC Relaxed (T=0)\nGrid: {L}x{L} | $a_x$={ax_sc_rel:.3f}, $a_y$={ay_sc_rel:.3f}",
        dot_size=30
    )
    im4 = plot_panel(
        axs[1, 1], spins_sc_int, Lx_sc_new, Ly_sc_new, ax_sc_new, ay_sc_new, 
        f"SC Interpolated (Standard $a$)\nGrid: {Lx_sc_new}x{Ly_sc_new} | $a_x$=$a_y$={a_com:.3f}",
        dot_size=10,
        show_ylabel=False
    )
    
    # Add a unified colorbar for the heatmaps at the bottom
    fig.subplots_adjust(bottom=0.15, top=0.90, left=0.08, right=0.92, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.025])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Out-of-plane spin component $n_z$", fontsize=10, fontweight='medium', labelpad=5)
    cbar.ax.tick_params(labelsize=8)
    
    # Main super title
    fig.suptitle(
        "Grid Interpolation Pipeline: Zero-Temperature Relaxation to Finite-Temperature SDE Input\n"
        r"Visualizing physical cell preservation under standardization of grid spacing ($a_{com} = " + f"{a_com:.2f}$)",
        fontsize=13, fontweight='bold', y=0.97
    )
    
    # Save the output figure
    out_dir = "output/plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "grid_interpolation_visualization.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved beautiful visualization to '{out_path}'")
    plt.close()

if __name__ == "__main__":
    main()
