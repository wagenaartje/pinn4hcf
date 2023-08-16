from fluidfoam import readmesh
from fluidfoam import readscalar, readvector
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

sol = './reference/'

# coordinates
x, y, z = readmesh(sol, structured=False,precision=10)

t = '5.9999944797'
# We want to extract: r, p, Ux, Uy
r = readscalar(sol, t, 'rho', False).T
p = readscalar(sol, t, 'p', False).T
U = readvector(sol, t, 'U', False).T
u = U[:,0]
v = U[:,1]

data = np.stack((r,p,u,v), axis=1)

fig,ax=plt.subplots(1)


# Plot colormap of mach number, only near airfoil
N = 256
X,Y = np.meshgrid(
    np.linspace(0.0, 2.0, N),
    np.linspace(0.0, 2.0, N)
) 


data = griddata((x,y), data, (X,Y), method='cubic')

# Create bezier curve
t = np.linspace(0,1,1000).reshape(-1,1)
p1 = np.array([0.5, 0.0]).reshape(-1,2)
p2 = np.array([1.0, 0.25]).reshape(-1,2)
p3 = np.array([2.0, 0.25]).reshape(-1,2)

B = (1-t)**2 * p1 + 2*(1-t)*t*p2 + t**2 * p3


def f (x):
    t_ind = np.argmin(np.abs(B[:,0] - x))
    return B[t_ind,1]

fvunc = np.vectorize(f)

# Remove part below curve
mapping = (Y > fvunc(X)) | (X < 0.5)
data[~mapping,:] = np.nan


if True:
    c = ax.pcolor(X,Y,data[:,:,1], cmap='rainbow')
    bar = fig.colorbar(c)
    bar.set_label(r'$p/p_{0,\infty}\ \mathrm{(-)}$', fontsize=12)

    ax.set_aspect(1)

    plt.show()

data = data.reshape(-1,4)

u_test = data
x_test = X.reshape(-1,1)
y_test = Y.reshape(-1,1)

# Save solution
d = {}
d['x_test'] = x_test
d['y_test'] = y_test
d['u_test'] = u_test
np.savez(f'./curved/reference/reference.npz', **d)
