''' Imports '''
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import utils

# Import the model architecture
sys.path.append('./detached')
import cpinn_phi_viscosity_neumann

# Load custom cmap
colors = np.load('cmap.npy')
cmap = clr.LinearSegmentedColormap.from_list("", colors)


''' Constants '''
torch.set_default_dtype(torch.float32)
device = torch.device('cpu')


''' Get reference solution '''
data = np.load(f'./detached/reference/reference.npz')
x_test = torch.tensor(data['x_test'], device=device, dtype=torch.float32)
y_test = torch.tensor(data['y_test'], device=device, dtype=torch.float32)
u_test = torch.tensor(data['u_test'], device=device, dtype=torch.float32)

# Convert to tuple
X = x_test.reshape(256, 256)
Y = y_test.reshape(256, 256)
x_test = (X.reshape(-1,1), Y.reshape(-1,1))


''' Define plotting function '''
def plot_heatmap (fig, ax, u):
    u = u.detach()
    u = u.reshape(256,256)

    # Exclude points inside the object and at the edges
    u[(u_test[:,0].isnan().reshape(256,256)) & (X >= 0.5) & (Y <= 0.25) ] = np.nan 
    
    # Plot the heatmap
    im = ax.pcolor(X, Y, u, cmap=cmap,  rasterized=True, linewidth=0)
    fig.colorbar(im, ax=ax)

    ax.set_aspect(1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot outer perimeter of object to hide "jagged" contour
    p1 = torch.tensor([0.5, 0.0], device=device).reshape(-1,2)
    p2 = torch.tensor([1.0, 0.25], device=device).reshape(-1,2)
    p3 = torch.tensor([2.0, 0.25], device=device).reshape(-1,2)
    t = torch.linspace(0,1,1000, device=device).reshape(-1,1)
    B = (1-t)**2 * p1 + 2*(1-t)*t*p2 + t**2 * p3

    ax.plot(B[:,0], B[:,1], 'white', linewidth=2)


''' Load model and do predictions '''
checkpoint = utils.get_checkpoint('./detached/cpinn_phi_viscosity_neumann.p', 'latest', truncate=False)

# Generator
generator = cpinn_phi_viscosity_neumann.Generator()
generator.load_state_dict(checkpoint['g_params'])
generator = generator.float()

# Make predictions
u_pred = generator(x_test)
r = u_pred[:,0]


''' Make plots'''
fig, ax = plt.subplots(1,1)
plot_heatmap(fig, ax, r)
plt.show()