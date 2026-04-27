import numpy as np
import numba as nb

@nb.njit
def get_energy_diff(spins, ix, iy, S_new, L, J, D, B, A):
    """
    Calculates the change in energy for a single spin flip.
    Includes: Exchange, DMI, Zeeman, and Anisotropy.
    """
    dS = S_new - spins[ix, iy]
    
    # Periodic Neighbors
    S_r = spins[(ix + 1) % L, iy]
    S_l = spins[(ix - 1 + L) % L, iy]
    S_u = spins[ix, (iy + 1) % L]
    S_d = spins[ix, (iy - 1 + L) % L]
    
    # Energy terms
    dE_ex = -J * np.dot(dS, S_r + S_l + S_u + S_d)
    
    # DMI Vector sum
    v_sum = np.array([
        D * (S_l[2] - S_r[2]), 
        D * (S_d[2] - S_u[2]), 
        D * (S_r[0] - S_l[0] + S_u[1] - S_d[1])
    ])
    dE_dmi = -np.dot(dS, v_sum)
    
    dE_Z = -B * dS[2]
    dE_A = A * (S_new[2]**2 - spins[ix, iy][2]**2)
    
    return dE_ex + dE_dmi + dE_Z + dE_A

@nb.njit
def mc_step(spins, L, J, D, B, A, T):
    """
    Performs L*L local Metropolis updates.
    """
    for _ in range(L * L):
        ix, iy = np.random.randint(0, L), np.random.randint(0, L)
        
        # Propose new spin vector
        S_new = spins[ix, iy] + (np.random.rand(3) - 0.5) * 0.5
        S_new /= np.linalg.norm(S_new)
        
        dE = get_energy_diff(spins, ix, iy, S_new, L, J, D, B, A)
        
        # Metropolis Criterion
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            spins[ix, iy] = S_new

def run_simulation(L=15, J=1.0, D=0.5, h_scaled=1.5, a_scaled=-0.5, T_start=1.0, T_target=0.01, steps=10000):
    """
    Standard Cooling Simulation:
    Starts from random spins and cools down to T_target.
    Used for initial state preparation.
    """
    B, A = h_scaled * (D**2 / J), a_scaled * (D**2 / J)
    
    # Random Initialization
    spins = np.zeros((L, L, 3))
    for i in range(L):
        for j in range(L):
            z = np.random.uniform(-1, 1)
            phi = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(1 - z**2)
            spins[i, j] = [r * np.cos(phi), r * np.sin(phi), z]

    anneal_limit = int(steps * 0.6)
    for step in range(steps):
        # Linear cool-down
        frac = max(0, (anneal_limit - step) / anneal_limit) if step < anneal_limit else 0
        current_T = T_target + (T_start - T_target) * frac
        
        mc_step(spins, L, J, D, B, A, current_T)
        
    return spins

def run_heating_step(spins, T_start, T_end, L=15, J=1.0, D=0.5, h_scaled=1.5, a_scaled=-0.5, steps=5000):
    """
    Heating Increment Protocol:
    1. Heats spins from T_start to T_end linearly over first 80% of steps.
    2. Relaxes at T_end for the final 20% of steps.
    
    Designed to be called iteratively for quasi-static heating loops.
    """
    B, A = h_scaled * (D**2 / J), a_scaled * (D**2 / J)
    
    # Ensure we don't overwrite the input array in a way that breaks parallelism
    current_spins = spins.copy()
    
    ramp_end_step = int(steps * 0.8)
    
    for step in range(steps):
        if step < ramp_end_step:
            # Linear heating ramp
            frac = step / ramp_end_step
            current_T = T_start + (T_end - T_start) * frac
        else:
            # Relaxation phase at target T
            current_T = T_end
            
        mc_step(current_spins, L, J, D, B, A, current_T)
        
    return current_spins