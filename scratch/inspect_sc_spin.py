import numpy as np

data = np.load('output/spin_states/fintemp/fintemp_relaxed_SC_super_T0.01_H0.1.npz')
spins = data['spins']
ax = data['ax']
ay = data['ay']

print(f"Shape of spins: {spins.shape}")
print(f"ax: {ax}, ay: {ay}")
print(f"n_x: min={np.min(spins[..., 0]):.6f}, max={np.max(spins[..., 0]):.6f}, mean={np.mean(spins[..., 0]):.6f}, var={np.var(spins[..., 0]):.6f}")
print(f"n_y: min={np.min(spins[..., 1]):.6f}, max={np.max(spins[..., 1]):.6f}, mean={np.mean(spins[..., 1]):.6f}, var={np.var(spins[..., 1]):.6f}")
print(f"n_z: min={np.min(spins[..., 2]):.6f}, max={np.max(spins[..., 2]):.6f}, mean={np.mean(spins[..., 2]):.6f}, var={np.var(spins[..., 2]):.6f}")
