# Skyrmion LLG Solver & Monte Carlo Simulator

This project contains a suite of high-performance Python tools for studying magnetic skyrmions. It includes an optimizer based on the deterministic Landau-Lifshitz-Gilbert (LLG) equation for exact phase stability analysis, finite-temperature SDE extensions for robust thermodynamic mapping, and a Monte Carlo simulator using the Metropolis algorithm to model the formation dynamics of magnetic skyrmion lattices.

## Physics Model
The classical spin Hamiltonian used to stabilize the skyrmions on a discrete 2D square lattice includes:
* **Heisenberg Exchange** ($J$): Ferromagnetic coupling between nearest neighbors.
* **Dzyaloshinskii-Moriya Interaction** ($D$): Interfacial antisymmetric exchange that favors chiral spin textures perpendicular to the neighbor direction.
* **Zeeman Energy** ($B$): Coupling to an external out-of-plane magnetic field.
* **Uniaxial Anisotropy** ($K$): Easy-axis out-of-plane anisotropy favoring $z$-oriented spins.

## Features
* **High-Performance LLG Solver**: Numerically integrates the overdamped Landau-Lifshitz-Gilbert (LLG) equation to accurately find exact theoretical magnetic ground states (SkX, SC, SP, FM). Uses a robust Heun (RK2) integrator with fully dynamic spatial scaling, natively compiled with Numba for ~100x speedups.
* **Finite-Temperature SDE Solver**: Leverages `JAX` and `Diffrax` to integrate stochastic differential equations (SDEs), enabling thermodynamic evaluations and rigorous average energy calculations of magnetic geometries at finite thermal gradients.
* **Topological Phase Diagrams**: Systematically sweep across applied magnetic fields and anisotropy boundaries to construct precise quantitative stability phase diagrams of topological magnetic states. This natively incorporates heavily optimized multicore processing parameters.
* **Monte Carlo Simulated Annealing**: Smoothly cools the system from a high-temperature random state to capture the dynamic thermal nucleation of a skyrmion lattice.
* **Live Visualization & Video Export**: Both numerical solvers feature real-time Matplotlib integrations utilizing multidimensional quiver plots, alongside automated MP4 video exports for monitoring structural formation.

## Dependencies
You can install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Running the Project

**1. Deterministic Phase Stability Extractor**
Generates and plots a full numerical phase diagram comparing the energy densities of various theoretical skyrmion phase configurations (Ansatzes) perfectly relaxed at `T=0`.
```bash
python phase_diagram.py --nH 26 --nA 33 --L 32
```

Focused high-precision window (anisotropy 1.0 to 1.5, magnetic field 0.0 to 0.5) at lattice size 128:
```bash
python phase_diagram.py --nH 51 --nA 51 --L 128 --Hmin 0.0 --Hmax 0.5 --Amin 1.0 --Amax 1.5 --recompute
```

**2. Finite-Temperature SDE Phase Mapping**
Runs stochastic SDE integrations mapping true thermodynamic energies arrayed cleanly across a full configuration grid at a specified base temperature.
```bash
python fintemp_phase_diagram.py --T 0.1 --steps 1000 --block 100 --workers 4
```

**3. Deterministic Point Relaxation**
Test a specific Hamiltonian parameter set by relaxing analytical ansatz formulations directly to local minimums.
```bash
python LLG_solver.py --H 1.0 --A 0.8 --L 64 --live-plot
```

**4. Finite-Temperature Single Point Analytics**
Simulate specific topological geometry structures and evaluate thermodynamic competition utilizing the direct stochastic framework.
```bash
python fintemp_LLG.py --L 32 --T 0.05 --H 1.0 --A 1.0 --steps 5000
```

**5. Monte Carlo Nucleation**
Simulate thermal melting and nucleation processes, plotting real-time visualizations.
```bash
python MC_metropolis.py
```

**6. Periodic Lattice Visualization**
Load `.npy` or `.npz` spin outputs from any of the local tools to meticulously visually analyze topological configurations tiled periodically across coordinate space.
```bash
python periodic_plotting.py "output/Fintemp/fintemp_groundstate_T0.05_H1.0.npz" --tiles 2 --mode quiver
```
