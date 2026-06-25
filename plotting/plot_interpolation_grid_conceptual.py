import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow imports from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLG_solver import init_SkX, init_SC
from fintemp_LLG import interpolate_spins_periodic

def main():
    print("=== Generating Conceptual Grid Interpolation Schematic ===")
    L = 6  # Small grid size for a highly legible, schematic relaxed grid
    a_com = 0.15  # Standardized spacing to yield clean, distinct grids on the right
    
    # 1. Initialize SkX Phase (Analytical Ansatz)
    print("[1/4] Generating idealized SkX ansatz...")
    spins_skx_init, ax_skx_init, ay_skx_init = init_SkX(L)
    P_x_skx, P_y_skx = L * ax_skx_init, L * ay_skx_init
    Lx_skx_new = max(1, int(round(P_x_skx / a_com)))
    Ly_skx_new = max(1, int(round(P_y_skx / a_com)))
    ax_skx_new = a_com
    ay_skx_new = a_com
    spins_skx_int = interpolate_spins_periodic(spins_skx_init, Lx_skx_new, Ly_skx_new)
    
    print(f"  SkX Relaxed (Concept): {L}x{L} @ ax={ax_skx_init:.3f}, ay={ay_skx_init:.3f} (Box: {P_x_skx:.3f} x {P_y_skx:.3f})")
    print(f"  SkX Interpolated: {Lx_skx_new}x{Ly_skx_new} @ ax={ax_skx_new:.3f}, ay={ay_skx_new:.3f}")
    
    # 2. Initialize SC Phase (Analytical Ansatz)
    print("[2/4] Generating idealized SC ansatz...")
    spins_sc_init, ax_sc_init, ay_sc_init = init_SC(L)
    P_x_sc, P_y_sc = L * ax_sc_init, L * ay_sc_init
    Lx_sc_new = max(1, int(round(P_x_sc / a_com)))
    Ly_sc_new = max(1, int(round(P_y_sc / a_com)))
    ax_sc_new = a_com
    ay_sc_new = a_com
    spins_sc_int = interpolate_spins_periodic(spins_sc_init, Lx_sc_new, Ly_sc_new)
    
    print(f"  SC Relaxed (Concept): {L}x{L} @ ax={ax_sc_init:.3f}, ay={ay_sc_init:.3f} (Box: {P_x_sc:.3f} x {P_y_sc:.3f})")
    print(f"  SC Interpolated: {Lx_sc_new}x{Ly_sc_new} @ ax={ax_sc_new:.3f}, ay={ay_sc_new:.3f}")
    
    # -------------------------------------------------------------
    # Plotting code
    # -------------------------------------------------------------
    print("\nGenerating conceptual schematic plot...")
    
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9.5), dpi=300)
    
    # helper function to plot a single panel conceptually
    def plot_panel(ax_plot, spins, L_x, L_y, ax_val, ay_val, title, dot_size=30, is_relaxed=True, show_ylabel=True):
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
            alpha=0.5,
            interpolation='bicubic'
        )
        
        # Plot grid points as black dots with a fine white border for maximum contrast
        ax_plot.scatter(
            X.flatten(), 
            Y.flatten(), 
            color='black', 
            edgecolors='white', 
            linewidths=0.5, 
            s=dot_size, 
            zorder=3
        )
        
        # Draw physical boundary box of the unit cell
        width = L_x * ax_val
        height = L_y * ay_val
        rect = plt.Rectangle(
            (-0.5 * ax_val, -0.5 * ay_val), 
            width, 
            height, 
            fill=False, 
            edgecolor='#111111', 
            linestyle='--', 
            linewidth=1.5, 
            alpha=0.9,
            zorder=2
        )
        ax_plot.add_patch(rect)
        
        # Draw conceptual dimension lines (W_x and W_y) to visually show cell conservation
        # Offset slightly outside the cell boundaries
        x_min, x_max = -0.5 * ax_val, -0.5 * ax_val + width
        y_min, y_max = -0.5 * ay_val, -0.5 * ay_val + height
        
        offset_y = 0.25 * height
        offset_x = 0.25 * width
        
        # Width Dimension Line (below the box)
        ax_plot.annotate(
            '', 
            xy=(x_min, y_min - offset_y), 
            xytext=(x_max, y_min - offset_y), 
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1, shrinkA=0, shrinkB=0),
            zorder=4
        )
        ax_plot.text(
            (x_min + x_max) / 2, 
            y_min - offset_y - 0.08 * height, 
            r'Physical Width $W_x$', 
            ha='center', 
            va='top', 
            fontsize=8.5, 
            color='#444444',
            fontweight='medium'
        )
        
        # Height Dimension Line (left of the box)
        ax_plot.annotate(
            '', 
            xy=(x_min - offset_x, y_min), 
            xytext=(x_min - offset_x, y_max), 
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1, shrinkA=0, shrinkB=0),
            zorder=4
        )
        ax_plot.text(
            x_min - offset_x - 0.08 * width, 
            (y_min + y_max) / 2, 
            r'Physical Height $W_y$', 
            ha='right', 
            va='center', 
            rotation=90, 
            fontsize=8.5, 
            color='#444444',
            fontweight='medium'
        )
        
        ax_plot.set_title(title, fontsize=11.5, fontweight='bold', pad=10)
        
        # Set aspect ratio to equal to avoid distortion of physical dimensions
        ax_plot.set_aspect('equal')
        
        # Set limits to accommodate dimension lines cleanly
        ax_plot.set_xlim(x_min - 0.45 * width, x_max + 0.15 * width)
        ax_plot.set_ylim(y_min - 0.45 * height, y_max + 0.15 * height)
        
        # Remove tick values to make it a generic conceptual schematic
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])
        
        # Add a clean text box explaining the grid properties
        if is_relaxed:
            props_str = f"Phase-Specific Grid\nSpacing: $a_x, a_y$\nGrid size: {L_x}$\\times${L_y}"
        else:
            props_str = f"Standardized Grid\nSpacing: $a_x = a_y = a_{{com}}$\nGrid size: {L_x}$\\times${L_y}"
            
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#dddddd')
        ax_plot.text(
            x_max - 0.02 * width, 
            y_max - 0.02 * height, 
            props_str, 
            transform=ax_plot.transData, 
            fontsize=8,
            verticalalignment='top', 
            horizontalalignment='right', 
            bbox=props,
            zorder=5
        )
        
        return im

    # Row 1: SkX Phase
    plot_panel(
        axs[0, 0], spins_skx_init, L, L, ax_skx_init, ay_skx_init, 
        "Skyrmion Crystal (SkX): Relaxed Unit Cell",
        dot_size=40, is_relaxed=True
    )
    plot_panel(
        axs[0, 1], spins_skx_int, Lx_skx_new, Ly_skx_new, ax_skx_new, ay_skx_new, 
        "Skyrmion Crystal (SkX): Standardized Grid",
        dot_size=12, is_relaxed=False, show_ylabel=False
    )
    
    # Row 2: SC Phase
    im3 = plot_panel(
        axs[1, 0], spins_sc_init, L, L, ax_sc_init, ay_sc_init, 
        "Square Cell Phase (SC): Relaxed Unit Cell",
        dot_size=40, is_relaxed=True
    )
    im4 = plot_panel(
        axs[1, 1], spins_sc_int, Lx_sc_new, Ly_sc_new, ax_sc_new, ay_sc_new, 
        "Square Cell Phase (SC): Standardized Grid",
        dot_size=12, is_relaxed=False, show_ylabel=False
    )
    
    # Add a unified colorbar for the heatmaps at the bottom
    fig.subplots_adjust(bottom=0.14, top=0.88, left=0.08, right=0.92, hspace=0.38, wspace=0.15)
    cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.02])
    cbar = fig.colorbar(im3, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Out-of-plane spin component $n_z$", fontsize=9.5, fontweight='medium', labelpad=4)
    cbar.ax.tick_params(labelsize=8)
    
    # Main super title
    fig.suptitle(
        "Concept Schematic: Grid Standardization for Finite-Temperature SDE Inputs\n"
        "Preserving physical cell dimensions while resampling different phases to a common spacing ($a_{com}$)",
        fontsize=12.5, fontweight='bold', y=0.96
    )
    
    # Save the output figures
    out_dir = "output/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as conceptual schematic
    path_conceptual = os.path.join(out_dir, "grid_interpolation_conceptual.png")
    plt.savefig(path_conceptual, dpi=300, bbox_inches='tight')
    
    # Also overwrite the visualization path so the user can easily load/see it
    path_visualization = os.path.join(out_dir, "grid_interpolation_visualization.png")
    plt.savefig(path_visualization, dpi=300, bbox_inches='tight')
    
    print(f"Saved conceptual schematic to '{path_conceptual}'")
    print(f"Overwrote visualization path '{path_visualization}'")
    plt.close()

if __name__ == "__main__":
    main()
