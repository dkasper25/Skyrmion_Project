import numpy as np
import matplotlib.pyplot as plt
import os
from LLG_solver import init_SkX, analyze_state

def test_morph_skx_to_square(steps=50, L=128):
    # Initialize pure SkX ansatz
    spins, ax_start, ay_start = init_SkX(L)
    
    # We want to artificially squish ax down to ay_start so the box becomes a square (Aspect Ratio = 1.0)
    ax_values = np.linspace(ax_start, ay_start, steps)
    
    c4_values = []
    c6_values = []
    classifications = []
    aspect_ratios = []
    
    print("Testing SkX artificial geometric morphing...")
    print(f"Starting Aspect Ratio: {ax_start/ay_start:.3f} (Ideal Hexagon)")
    print(f"Ending Aspect Ratio: {ay_start/ay_start:.3f} (Perfect Square)")
    
    for ax_current in ax_values:
        aspect = ax_current / ay_start
        aspect_ratios.append(aspect)
        
        # Analyze the state using the artificially squished grid dimensions
        # We do NOT run relax_phase. We just run the math classifier on the stretched grid.
        stats = analyze_state(spins, ax_current, ay_start, phase_name=f"Morph_{aspect:.2f}", plot_fft=False)
        
        c4_values.append(stats['c4'])
        c6_values.append(stats['c6'])
        classifications.append(stats['classified_state'])
        
        print(f"Aspect Ratio: {aspect:.3f} | C4: {stats['c4']:.3f} | C6: {stats['c6']:.3f} | Classification: {stats['classified_state']}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Artificial Grid Aspect Ratio (ax / ay)')
    ax1.set_ylabel('Symmetry Strength ($C_m$)')
    
    # Note: We reverse the x-axis so it reads from 1.73 down to 1.0
    ax1.plot(aspect_ratios, c6_values, marker='h', color='green', label='C6 (Hexagonal)')
    ax1.plot(aspect_ratios, c4_values, marker='s', color='orange', label='C4 (Square)')
    
    ax1.axvline(np.sqrt(3), color='gray', linestyle='--', alpha=0.5)
    ax1.text(np.sqrt(3)+0.01, 0.5, 'Ideal SkX Box', rotation=90, color='gray')
    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.text(1.01, 0.5, 'Square Box', rotation=90, color='gray')

    # Shade background based on classification
    for i in range(len(aspect_ratios) - 1):
        color = 'lightblue' if classifications[i] == 'SkX' else ('lightgray')
        ax1.axvspan(aspect_ratios[i], aspect_ratios[i+1], color=color, alpha=0.5, linewidth=0)

    ax1.legend()
    plt.title("Pure SkX Ansatz: Artificial Geometric Squishing Test")
    plt.grid(True, alpha=0.3)
    
    # Invert x-axis to visually match "morphing from SkX down to Square"
    ax1.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('output/LLG/Graphs/SkX_Morph_Test.png', dpi=150)
    print("\nSaved graph to 'output/LLG/Graphs/SkX_Morph_Test.png'")

if __name__ == "__main__":
    os.makedirs("output/LLG/Graphs", exist_ok=True)
    test_morph_skx_to_square()
