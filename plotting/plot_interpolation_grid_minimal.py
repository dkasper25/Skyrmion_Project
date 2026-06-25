import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# Allow imports from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLG_solver import init_SkX, init_SC

def main():
    print("=== Generating Minimal Conceptual Grid Interpolation Schematic ===")
    L = 6  # Small grid size for relaxed grid
    a_com = 0.20  # Standardized spacing
    
    # SkX Grid Parameters
    _, ax_skx, ay_skx = init_SkX(L)
    P_x_skx, P_y_skx = L * ax_skx, L * ay_skx
    Lx_skx_new = max(1, int(round(P_x_skx / a_com)))
    Ly_skx_new = max(1, int(round(P_y_skx / a_com)))
    
    # SC Grid Parameters
    _, ax_sc, ay_sc = init_SC(L)
    P_x_sc, P_y_sc = L * ax_sc, L * ay_sc
    Lx_sc_new = max(1, int(round(P_x_sc / a_com)))
    Ly_sc_new = max(1, int(round(P_y_sc / a_com)))
    
    # Setup Figure with 3 columns: Left (Relaxed), Middle (Arrow), Right (Standardized)
    fig, axs = plt.subplots(2, 3, figsize=(8.0, 5.2), 
                            gridspec_kw={'width_ratios': [1.0, 0.25, 1.0], 'hspace': 0.35, 'wspace': 0.10}, 
                            dpi=300)
    
    def plot_grid(ax_plot, grid_Lx, grid_Ly, grid_ax, grid_ay, phys_Px, phys_Py, dot_size=45):
        # Grid coordinates
        x = np.arange(grid_Lx) * grid_ax
        y = np.arange(grid_Ly) * grid_ay
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Plot grid points
        ax_plot.scatter(X.flatten(), Y.flatten(), color='black', s=dot_size, zorder=3)
        
        # Draw physical boundary box of the unit cell (matching original physical dimensions)
        rect = plt.Rectangle((0, 0), phys_Px, phys_Py, fill=False, edgecolor='black', linestyle='--', linewidth=1.5, zorder=2)
        ax_plot.add_patch(rect)
        
        # Set aspect ratio to equal to avoid distortion of physical dimensions
        ax_plot.set_aspect('equal')
        
        # Set limits with a clean margin around the original physical box
        ax_plot.set_xlim(-0.1 * phys_Px, 1.1 * phys_Px)
        ax_plot.set_ylim(-0.1 * phys_Py, 1.1 * phys_Py)
        
        # Remove axis ticks/numbers entirely
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])
        
        # Hide the solid black border box (spines) around the grid
        for spine in ax_plot.spines.values():
            spine.set_visible(False)
        
    def plot_arrow(ax_plot):
        # Hide all axes
        ax_plot.axis('off')
        # Draw a horizontal arrow in the center using a sharp geometric block arrow
        ax_plot.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5), xycoords='axes fraction',
                         arrowprops=dict(facecolor='black', edgecolor='black', width=2.5, headwidth=9, headlength=9, shrink=0.05))
        # Add a text label "Interpolate" above the arrow
        ax_plot.text(0.5, 0.55, 'Interpolate', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', transform=ax_plot.transAxes)

    # Row 0: SkX
    plot_grid(axs[0, 0], L, L, ax_skx, ay_skx, P_x_skx, P_y_skx, dot_size=32)
    plot_arrow(axs[0, 1])
    plot_grid(axs[0, 2], Lx_skx_new, Ly_skx_new, a_com, a_com, P_x_skx, P_y_skx, dot_size=8)
    axs[0, 0].set_title(f"Relaxed Grid\n$a_x = {ax_skx:.3f}, a_y = {ay_skx:.3f}$", fontsize=11, fontweight='bold')
    axs[0, 2].set_title(f"Standardized Grid\n$a_x = a_y = {a_com:.3f}$", fontsize=11, fontweight='bold')
    
    # Label the whole SkX line on the left (perfectly aligned horizontally across both rows using blended transform)
    trans0 = mtransforms.blended_transform_factory(fig.transFigure, axs[0, 0].transAxes)
    axs[0, 0].text(0.06, 0.5, "SkX", transform=trans0, 
                  ha='right', va='center', fontsize=12, fontweight='bold', color='black')
    
    # Row 1: SC
    plot_grid(axs[1, 0], L, L, ax_sc, ay_sc, P_x_sc, P_y_sc, dot_size=32)
    plot_arrow(axs[1, 1])
    plot_grid(axs[1, 2], Lx_sc_new, Ly_sc_new, a_com, a_com, P_x_sc, P_y_sc, dot_size=8)
    
    axs[1, 0].set_title(f"Relaxed Grid\n$a_x = {ax_sc:.3f}, a_y = {ay_sc:.3f}$", fontsize=11, fontweight='bold')
    axs[1, 2].set_title(f"Standardized Grid\n$a_x = a_y = {a_com:.3f}$", fontsize=11, fontweight='bold')
    
    # Label the whole SC line on the left (perfectly aligned horizontally across both rows using blended transform)
    trans1 = mtransforms.blended_transform_factory(fig.transFigure, axs[1, 0].transAxes)
    axs[1, 0].text(0.06, 0.5, "SC", transform=trans1, 
                  ha='right', va='center', fontsize=12, fontweight='bold', color='black')
    
    # Save the output figures
    out_dir = "output/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    path_minimal = os.path.join(out_dir, "grid_interpolation_minimal.png")
    plt.savefig(path_minimal, dpi=300, bbox_inches='tight')
    
    path_visualization = os.path.join(out_dir, "grid_interpolation_visualization.png")
    plt.savefig(path_visualization, dpi=300, bbox_inches='tight')
    
    print(f"Saved minimal schematic to '{path_minimal}'")
    print(f"Overwrote visualization path '{path_visualization}'")
    plt.close()

if __name__ == "__main__":
    main()
