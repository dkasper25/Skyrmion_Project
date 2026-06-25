import os
import sys
# Allow imports from the parent directory (project root) when running this script directly or as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from LLG_solver import init_SkX, init_SC, relax_phase, analyze_state

def test_transition(H_fixed=1.2, A_start=1.0, A_end=1.7, steps=20, L=128):
    A_values = np.linspace(A_start, A_end, steps)
    
    # Track energies
    e_skx_list, e_sc_list = [], []
    
    # Track SkX metrics
    skx_aspect = []
    skx_c4 = []
    skx_c6 = []
    
    # Track SC metrics
    sc_aspect = []
    sc_c4 = []
    sc_c6 = []
    
    # Track Ground State winner
    gs_classifications = []
    
    print(f"Testing SkX vs SC Ground State Transition over A = {A_start} to {A_end} at H = {H_fixed}")
    
    for A in A_values:
        print(f"\n--- Testing A = {A:.2f} ---")
        
        # 1. Test SkX Ansatz
        spins_skx, ax_in_skx, ay_in_skx = init_SkX(L)
        relaxed_skx, E_skx, ax_skx, ay_skx = relax_phase(
            spins_skx, L, H_scaled=H_fixed, A_scaled=A, phase_name="SkX_test", 
            ax_in=ax_in_skx, ay_in=ay_in_skx, max_steps=10000, tol=1e-6
        )
        stats_skx = analyze_state(relaxed_skx, ax_skx, ay_skx, phase_name=f"SkX_A{A:.2f}", plot_fft=False)
        a_skx = max(ax_skx, ay_skx) / min(ax_skx, ay_skx)
        
        e_skx_list.append(E_skx)
        skx_aspect.append(a_skx)
        skx_c4.append(stats_skx['c4'])
        skx_c6.append(stats_skx['c6'])
        
        # 2. Test SC Ansatz
        spins_sc, ax_in_sc, ay_in_sc = init_SC(L)
        relaxed_sc, E_sc, ax_sc, ay_sc = relax_phase(
            spins_sc, L, H_scaled=H_fixed, A_scaled=A, phase_name="SC_test", 
            ax_in=ax_in_sc, ay_in=ay_in_sc, max_steps=10000, tol=1e-6
        )
        stats_sc = analyze_state(relaxed_sc, ax_sc, ay_sc, phase_name=f"SC_A{A:.2f}", plot_fft=False)
        a_sc = max(ax_sc, ay_sc) / min(ax_sc, ay_sc)
        
        e_sc_list.append(E_sc)
        sc_aspect.append(a_sc)
        sc_c4.append(stats_sc['c4'])
        sc_c6.append(stats_sc['c6'])
        
        # 3. Determine the Ground State (lowest energy)
        if E_skx < E_sc:
            winner = "SkX Ansatz"
            gs_classifications.append(stats_skx['classified_state'])
        else:
            winner = "SC Ansatz"
            gs_classifications.append(stats_sc['classified_state'])
            
        print(f"Winner: {winner} (E = {min(E_skx, E_sc):.5f})")

    # --- Plotting the Transition ---
    fig, (ax_en, ax_ar, ax_sym) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Top Plot: Energies
    ax_en.set_title(f'Ground State Energy Competition (H={H_fixed})')
    ax_en.plot(A_values, e_skx_list, marker='o', color='blue', label='SkX Ansatz Energy')
    ax_en.plot(A_values, e_sc_list, marker='s', color='orange', label='SC Ansatz Energy')
    ax_en.set_ylabel('Energy Density')
    ax_en.legend(loc='upper right')
    ax_en.grid(True, alpha=0.3)

    # Middle Plot: Aspect Ratios
    ax_ar.set_title('Physical Aspect Ratios (max/min)')
    ax_ar.plot(A_values, skx_aspect, marker='h', color='blue', linestyle='--', label='SkX Ansatz AR')
    ax_ar.plot(A_values, sc_aspect, marker='s', color='orange', linestyle='--', label='SC Ansatz AR')
    ax_ar.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax_ar.axhline(np.sqrt(3), color='gray', linestyle=':', alpha=0.5)
    ax_ar.text(A_start, np.sqrt(3)+0.02, r'Ideal SkX ($\sqrt{3}$)', color='gray')
    ax_ar.text(A_start, 1.02, 'Ideal SC (1.0)', color='gray')
    ax_ar.set_ylabel('Aspect Ratio')
    ax_ar.legend(loc='upper right')
    ax_ar.grid(True, alpha=0.3)

    # Bottom Plot: Symmetry Strengths
    ax_sym.set_title('Angular Symmetry Strengths (C4 vs C6)')
    ax_sym.plot(A_values, skx_c6, marker='h', color='blue', label='SkX Ansatz (C6)', alpha=0.7)
    ax_sym.plot(A_values, skx_c4, marker='s', color='blue', linestyle=':', label='SkX Ansatz (C4)', alpha=0.7)
    
    ax_sym.plot(A_values, sc_c6, marker='h', color='orange', label='SC Ansatz (C6)', alpha=0.7)
    ax_sym.plot(A_values, sc_c4, marker='s', color='orange', linestyle=':', label='SC Ansatz (C4)', alpha=0.7)
    
    ax_sym.set_xlabel('Anisotropy (A)')
    ax_sym.set_ylabel('Strength ($C_m$)')
    ax_sym.legend(loc='upper right')
    ax_sym.grid(True, alpha=0.3)
    
    # Shade background of all plots based on final classification
    for i in range(len(A_values) - 1):
        color = 'lightblue' if gs_classifications[i] == 'SkX' else ('lightsalmon' if gs_classifications[i] == 'SC' else 'lightgray')
        ax_en.axvspan(A_values[i], A_values[i+1], color=color, alpha=0.5, linewidth=0)
        ax_ar.axvspan(A_values[i], A_values[i+1], color=color, alpha=0.5, linewidth=0)
        ax_sym.axvspan(A_values[i], A_values[i+1], color=color, alpha=0.5, linewidth=0)

    plt.tight_layout()
    plt.savefig(f'output/plots/llg/SkX_SC_Energy_Transition_Test_h={H_fixed}.png', dpi=150)
    print(f"\nTest complete! Saved graph to 'output/plots/llg/SkX_SC_Energy_Transition_Test_h={H_fixed}.png'")

if __name__ == "__main__":
    os.makedirs("output/plots/llg", exist_ok=True)
    test_transition()
