import numpy as np
import numba as nb

@nb.njit
def initialize_lattice(L):
    lattice = np.random.normal(0, 1, (L, L, 3))
    for i in range(L):
        for j in range(L):
            lattice[i, j] /= np.linalg.norm(lattice[i, j])
    return lattice

@nb.njit
def calculate_energy(lattice, L, J, D, B, A):
    energy = 0.0
    for ix in range(L):
        for iy in range(L):
            S = lattice[ix, iy]
            S_r = lattice[(ix + 1) % L, iy]
            S_u = lattice[ix, (iy + 1) % L]
            energy += -J * np.dot(S, S_r + S_u)
            energy += -D * (S[0]*S_u[2] - S[2]*S_u[0] + S_r[2]*S[1] - S_r[1]*S[2])
            energy += -B * S[2] + A * (S[2]**2)
    return energy / (L * L)

@nb.njit
def mc_step(lattice, L, J, D, B, A, T):
    for _ in range(L * L):
        ix, iy = np.random.randint(0, L), np.random.randint(0, L)
        S_old = lattice[ix, iy]
        S_new = S_old + (np.random.rand(3) - 0.5) * 0.5
        S_new /= np.linalg.norm(S_new)
        dS = S_new - S_old
        S_r, S_l = lattice[(ix + 1) % L, iy], lattice[(ix - 1 + L) % L, iy]
        S_u, S_d = lattice[ix, (iy + 1) % L], lattice[ix, (iy - 1 + L) % L]
        dE = -J * np.dot(dS, S_r + S_l + S_u + S_d) + \
             -np.dot(dS, np.array([D*(S_l[2]-S_r[2]), D*(S_d[2]-S_u[2]), D*(S_r[0]-S_l[0]+S_u[1]-S_d[1])])) + \
             -B * dS[2] + A * (S_new[2]**2 - S_old[2]**2)
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            lattice[ix, iy] = S_new
    return lattice

def run_video_protocol(L, h, a, T_START, T_MIN, T_MAX, steps_cool, steps_relax, steps_heat, frame_every):
    J, D = 1.0, 0.5
    B = h * (D**2 / J)
    A = a * (D**2 / J)
    lattice = initialize_lattice(L)
    frames = []

    # 1. COOLING
    for s in range(steps_cool):
        T = T_START + (T_MIN - T_START) * (s / steps_cool)
        lattice = mc_step(lattice, L, J, D, B, A, T)
        if s % 10000 == 0:
            print(f"Phase 1 (Cooling): {s}/{steps_cool} steps ({100*s/steps_cool:.1f}%)")
        if s % frame_every == 0:
            frames.append((lattice.copy(), T, calculate_energy(lattice, L, J, D, B, A)))
            
    # 2. ANNEAL
    for s in range(steps_relax):
        lattice = mc_step(lattice, L, J, D, B, A, T_MIN)
        if s % 10000 == 0:
            print(f"Phase 2 (Annealing): {s}/{steps_relax} steps ({100*s/steps_relax:.1f}%)")
        if s % frame_every == 0:
            frames.append((lattice.copy(), T_MIN, calculate_energy(lattice, L, J, D, B, A)))
            
    # 3. HEATING
    for s in range(steps_heat):
        T = T_MIN + (T_MAX - T_MIN) * (s / steps_heat)
        lattice = mc_step(lattice, L, J, D, B, A, T)
        if s % 10000 == 0:
            print(f"Phase 3 (Heating): {s}/{steps_heat} steps ({100*s/steps_heat:.1f}%)")
        if s % frame_every == 0:
            frames.append((lattice.copy(), T, calculate_energy(lattice, L, J, D, B, A)))
            
    return frames