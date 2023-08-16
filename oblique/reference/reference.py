# Source: modern compressible flow, Anderson, 4th ed, Section 4.3

''' Imports '''
import numpy as np
from scipy import optimize


''' Constants '''
# Physical constants
gamma = 1.4
p1 = 1.0
r1 = 1.0
M1 = 2.0
theta = np.deg2rad(10)

# Geometry
x_min = 0.0
x_max = 1.5
y_min = 0.0
y_max = 1.0
w_start = 0.5


''' Derived quantities '''
# Speed of sound
c1 = np.sqrt(gamma*p1/r1)
print(f'c1 = {c1:.3f}')

# Velocity
u1 = M1 * c1
print(f'u1 = {u1:.3f}')

# Mach angle
mu = np.arcsin(1/M1)
print(f'mu = {np.rad2deg(mu):.2f}')


''' Wave angles '''
def residual (beta):
    lhs = np.tan(theta)
    rhs = 2 * (1/np.tan(beta)) * (M1**2 * np.sin(beta)**2 - 1) / (M1**2 * (gamma + np.cos(2*beta)) + 2)

    return lhs - rhs

# Start search at 0 degrees and 90 degrees (outer bounds)
betas = optimize.fsolve(residual, [0.01, np.pi/2])
beta = betas[0]
print(f'beta = {np.rad2deg(beta):.3f}')


''' Oblique shock relations '''
r2 = r1 * ((gamma + 1) * M1**2 * np.sin(beta)**2) / ((gamma-1) * M1**2 * np.sin(beta)**2 + 2)
p2 = p1 * (1+(2*gamma)/(gamma+1) * (M1**2 * np.sin(beta)**2 - 1))
M2 = 1/ (np.sin(beta - theta)) * np.sqrt((M1**2 * np.sin(beta)**2 + 2/(gamma-1))/((2*gamma/(gamma-1)) * M1**2 * np.sin(beta)**2 - 1))
# T2 = T1 * (p2/p1) * (r1/r2)

# Horizontal and vertical velocity components
M2_u = M2 * np.cos(theta)
M2_v = M2 * np.sin(theta)

c2 = np.sqrt(gamma*p2/r2)

print(f'r2 = {r2:.3f}')
print(f'p2 = {p2:.3f}')
print(f'M2_tot = {M2:.3f}') # This is horizontal + vertical Mach number
print(f'M2_u = {M2_u:.3f}')
print(f'M2_v = {M2_v:.3f}')
print(f'c2 = {c2:.3f}')

print(f'u2 = {M2_u * c2:.3f}')
print(f'v2 = {M2_v * c2:.3f}')


''' Generate reference data '''
x = np.linspace(x_min, x_max, 256)
y = np.linspace(y_min, y_max, 256)
x, y = np.meshgrid(x, y)

x = x.flatten()
y = y.flatten()

# Remove elements outside of the wedge
mapping = y > (x-w_start)*np.tan(theta) + y_min
x = x[mapping]
y = y[mapping]

# Set reference solution
r = r1 * np.ones_like(x, dtype=np.float64)
p = p1 * np.ones_like(x, dtype=np.float64)
u = u1 * np.ones_like(x, dtype=np.float64)
v = np.zeros_like(x, dtype=np.float64)

mapping = y <= (x-w_start)*np.tan(beta) + y_min
r[mapping] = r2
p[mapping] = p2
u[mapping] = M2_u * c2
v[mapping] = M2_v * c2

u_test = np.vstack((r,p,u,v)).T
x_test = x.reshape(-1,1)
y_test = y.reshape(-1,1)

# Save solution
d = {}
d['x_test'] = x_test.astype(np.float32)
d['y_test'] = y_test.astype(np.float32)
d['u_test'] = u_test.astype(np.float32)
np.savez('./oblique/reference/reference.npz', **d)