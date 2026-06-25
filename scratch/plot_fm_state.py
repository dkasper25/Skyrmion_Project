import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from fintemp_LLG import compare_fintemp_phases

class Args:
    L = 32
    L_super = 64
    H = 0.1
    A = 0.25
    T = 0.01
    dt = 0.005
    steps = 1000
    block = 50
    seed = 42
    no_plot = True
    live_mode = "quiver"
    dynamic_scaling = False
    plot_fft = True # This will save the FFT plot
    save_all = True
    iso_scale = True
    standard_a = 0.2

args = Args()
winner, results, results_terms = compare_fintemp_phases(args)
