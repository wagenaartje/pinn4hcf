''' Imports'''
import torch
import time
import skopt
import pickle
import math
import numpy as np
import torch_cgd


''' Constants '''
# Physical constants
gamma = 1.4
r_inf = 1.0
p_inf = 1.0
M_inf = 2.0
v_inf = 0.0
u_inf = math.sqrt(gamma*p_inf/r_inf)*M_inf

# Geometry
x_min = 0.0
x_max = 1.5
y_min = 0.0
y_max = 1.0
w_angle = math.radians(10)
w_start = 0.5

# Hyperparameters
num_ib = 750
num_r = 10000


''' Partial differential equations '''
def pde (state, coords):
    # Coordinates
    x = coords[0]
    y = coords[1]

    # State
    r = state[:, [0]]
    p = state[:, [1]]
    u = state[:, [2]]
    v = state[:, [3]]
    
    rE = p/(gamma - 1) +0.5*r*(u**2+v**2)

    f1 = r*u
    f2 = r*u*u+p
    f3 = r*u*v
    f4 = (rE+p)*u
    
    g1 = r*v
    g2 = r*v*u
    g3 = r*v*v + p
    g4 = (rE+p)*v
    
    f1_x = torch.autograd.grad(f1, x, grad_outputs=torch.ones_like(f1),create_graph=True)[0]
    f2_x = torch.autograd.grad(f2, x, grad_outputs=torch.ones_like(f2),create_graph=True)[0]
    f3_x = torch.autograd.grad(f3, x, grad_outputs=torch.ones_like(f3),create_graph=True)[0]
    f4_x = torch.autograd.grad(f4, x, grad_outputs=torch.ones_like(f4),create_graph=True)[0]

    g1_y = torch.autograd.grad(g1, y, grad_outputs=torch.ones_like(g1),create_graph=True)[0]
    g2_y = torch.autograd.grad(g2, y, grad_outputs=torch.ones_like(g2),create_graph=True)[0]
    g3_y = torch.autograd.grad(g3, y, grad_outputs=torch.ones_like(g3),create_graph=True)[0]
    g4_y = torch.autograd.grad(g4, y, grad_outputs=torch.ones_like(g4),create_graph=True)[0]

    r1 = f1_x + g1_y
    r2 = f2_x + g2_y
    r3 = f3_x + g3_y
    r4 = f4_x + g4_y

    return r1, r2, r3, r4


''' Boundary conditions'''
def collocation_points (device='cpu'):
    # Left boundary
    x_l = torch.full((num_ib, 1), x_min, device=device)
    y_l = torch.linspace(y_min, y_max, num_ib, device=device).reshape(-1,1)

    u_l = (r_inf, p_inf, u_inf, v_inf)

    # Top boundary
    x_t = torch.linspace(x_min, x_max, num_ib, device=device).reshape(-1,1)
    y_t = torch.full((num_ib, 1), y_max, device=device)

    t_t = torch.full((num_ib,1), -torch.pi/2,device=device)

    # Bottom boundary
    x_b = torch.linspace(x_min,w_start, num_ib+1, device=device)[:-1].reshape(-1,1)
    y_b = torch.full((num_ib, 1), y_min, device=device)

    t_b = torch.full((num_ib,1), torch.pi/2, device=device)

    # Wedge boundary
    x_w = torch.linspace(w_start, x_max, num_ib, device=device).reshape(-1,1)
    y_w = y_min + (x_w-w_start)*math.tan(w_angle)

    t_w = torch.full((num_ib,1), torch.pi/2 + w_angle, device=device)

    # Residual points
    hammersley = skopt.sampler.Hammersly()
    x_r = torch.tensor(hammersley.generate([(x_min,x_max), (y_min, y_max)], num_r, random_state=0), device=device)
    mapping = x_r[:,1] > (x_r[:,0]-w_start)*math.tan(w_angle) + y_min
    x_r = x_r[mapping,:]

    # Convert to tuples
    x_l = (x_l, y_l)
    x_t = (x_t, y_t)
    x_b = (x_b, y_b)
    x_w = (x_w, y_w)
    x_r = (x_r[:,[0]], x_r[:,[1]])

    return x_l, u_l, x_t, t_t, x_b, t_b, x_w, t_w, x_r


''' Reference data '''
def reference (device='cpu'):
    data = np.load('./oblique/reference/reference.npz')
    x_test = torch.tensor(data['x_test'], device=device, dtype=torch.float64)
    y_test = torch.tensor(data['y_test'], device=device, dtype=torch.float64)
    u_test = torch.tensor(data['u_test'], device=device, dtype=torch.float64)

    # Convert to tuple
    x_test = (x_test, y_test)

    return x_test, u_test


''' Plot points '''
# import matplotlib.pyplot as plt

# x_l, u_l, x_t, t_t, x_b, t_b, x_w, t_w, x_r = collocation_points()
# x_test, u_test = reference()

# plt.scatter(x_l[0], x_l[1], label='Left boundary',zorder=6)
# plt.scatter(x_t[0], x_t[1], label='Top boundary', zorder=6)
# plt.scatter(x_b[0], x_b[1], label='Bottom boundary',zorder=6)
# plt.scatter(x_w[0], x_w[1], label='Wedge boundary', zorder=5)
# plt.scatter(x_r[0], x_r[1], label='Residual points',zorder=4)
# plt.scatter(x_test[0], x_test[1], label='Reference points',zorder=3)
# plt.legend()
# plt.gca().set_aspect('equal')
# plt.show()

# raise BaseException('Stop')


''' Model definition '''
# Generator
class Generator (torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Generate with float32 to use same initialization
        torch.manual_seed(0)
        self.map = torch.nn.Sequential(
            torch.nn.Linear( 2, 50, dtype=torch.float32), torch.nn.Tanh(),
            torch.nn.Linear(50, 50, dtype=torch.float32), torch.nn.Tanh(),
            torch.nn.Linear(50, 50, dtype=torch.float32), torch.nn.Tanh(),
            torch.nn.Linear(50, 50, dtype=torch.float32), torch.nn.Tanh(),
            torch.nn.Linear(50,  4, dtype=torch.float32)
        ).double()


    def forward (self, a: tuple) -> torch.Tensor:
        # Force positive density and pressure
        res = self.map(torch.cat(a, 1))
        res[:,0] = torch.exp(res[:,0])
        res[:,1] = torch.exp(res[:,1])
        return res

class Discriminator (torch.nn.Module):
    def __init__ (self):
        super().__init__()

         # Generate with float32 to use same initialization
        torch.manual_seed(1)
        self.map = torch.nn.Sequential(
            torch.nn.Linear(2, 50, dtype=torch.float32), torch.nn.ReLU(),
            torch.nn.Linear(50, 50, dtype=torch.float32), torch.nn.ReLU(),
            torch.nn.Linear(50, 50, dtype=torch.float32), torch.nn.ReLU(),
            torch.nn.Linear(50, 50, dtype=torch.float32), torch.nn.ReLU(),
            torch.nn.Linear(50, 50, dtype=torch.float32), torch.nn.ReLU(),
            torch.nn.Linear(50, 9, dtype=torch.float32)
        ).double()

    def forward (self, a: tuple) -> torch.Tensor:
        return self.map(torch.cat(a, 1))


def loss_slip (coords, thetas):
    state = generator(coords)
    disc = discriminator(coords)

    u = state[:, [2]]
    v = state[:, [3]]
    
    cos = torch.cos(thetas)
    sin = torch.sin(thetas)

    u_angle = u * cos + v * sin

    fb = (disc[:,[8]] * u_angle).mean()

    mse = (u_angle**2).mean().item()
    
    return fb, mse


# Loss function for PDE
def loss_pde (coords):
    global mse_pde, res_r, res_p, res_u, res_v

    for x in coords: x.requires_grad = True
    state = generator(coords)
    r1, r2, r3, r4 = pde(state, coords)

    for x in coords: x.requires_grad = False
    disc = discriminator(coords)

    f = (disc[:,[0]] * r1).mean() + \
        (disc[:,[1]] * r2).mean() + \
        (disc[:,[2]] * r3).mean() + \
        (disc[:,[3]] * r4).mean()
    
    res_r = (r1**2).mean().item()
    res_p = (r2**2).mean().item()
    res_u = (r3**2).mean().item()
    res_v = (r4**2).mean().item()

    mse_pde = res_r + res_p + res_u + res_v
        
    return f


def loss_inlet (x_bc, u_bc):
    u_pred = generator(x_bc)
    disc = discriminator(x_bc)
    
    loss = (disc[:,4] * (u_pred[:,0] - u_bc[0])).mean() + \
           (disc[:,5] * (u_pred[:,1] - u_bc[1])).mean() + \
           (disc[:,6] * (u_pred[:,2] - u_bc[2])).mean() + \
           (disc[:,7] * (u_pred[:,3] - u_bc[3])).mean()
    
    mse = ((u_pred[:,0] - u_bc[0])**2).mean().item() + \
           ((u_pred[:,1] - u_bc[1])**2).mean().item() + \
           ((u_pred[:,2] - u_bc[2])**2).mean().item() + \
           ((u_pred[:,3] - u_bc[3])**2).mean().item()
    
    return loss, mse


def evaluate ():
    global mse_bc, closs_bc, closs_r
    optimizer.zero_grad()                     

    loss_r = loss_pde(x_r)        
    loss_b, mse_b = loss_slip(x_b, t_b)
    loss_l, mse_l = loss_inlet(x_l, u_l)
    loss_c, mse_c = loss_slip(x_w, t_w) 

    mse_bc = mse_c + mse_l + mse_b

    loss_bc = loss_c + loss_l + loss_b
    loss = loss_r + loss_bc

    closs_bc = loss_bc.item()
    closs_r = loss_r.item()

    return loss

if __name__ == '__main__':
    # Set up hardware 
    device = torch.device('cuda')
    torch.set_default_dtype(torch.float64)

    # Load collocation points and data
    x_l, u_l, x_t, t_t, x_b, t_b, x_w, t_w, x_r = collocation_points(device)
    x_test, u_test = reference(device)

    # Set up models and optimizer
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    solver = torch_cgd.solvers.GMRES(tol=1e-7, atol=1e-20)
    optimizer = torch_cgd.ACGD(generator.parameters(), discriminator.parameters(), lr=0.001, solver=solver, eps=1e-8, beta=0.99)

    # Prepare logging
    checkpoint_file = open(f'./oblique/cpinn.p', 'ab')

    # Start looping
    epoch = 0
    start_time = time.time()

    while True:
        epoch_start_time = time.time() 

        # Optimize loss function
        loss = evaluate()
        if epoch > 0:
            optimizer.step(loss)

        # Calculate L2 error
        u_pred = generator(x_test)

        L2_r = torch.linalg.norm(u_pred[:,0] - u_test[:,0],2) / torch.linalg.norm(u_test[:,0],2)
        L2_p = torch.linalg.norm(u_pred[:,1] - u_test[:,1],2) / torch.linalg.norm(u_test[:,1],2)
        L2_u = torch.linalg.norm(u_pred[:,2] - u_test[:,2],2) / torch.linalg.norm(u_test[:,2],2)
        L2_v = torch.linalg.norm(u_pred[:,3] - u_test[:,3],2) / torch.linalg.norm(u_test[:,3],2)
        
        # Store data
        data = { 
            'epoch': epoch, 
            'time': time.time() - start_time, 
            'epoch_time': time.time() - epoch_start_time,
            'loss': loss.item(), 
            'loss_r': mse_pde,
            'loss_bc': mse_bc,
            'L2_r': L2_r.item(),
            'L2_p': L2_p.item(),
            'L2_u': L2_u.item(),
            'L2_v': L2_v.item(),
            'res_r': res_r,
            'res_p': res_p,
            'res_u': res_u,
            'res_v': res_v,
            'closs_r': closs_r,
            'closs_bc': closs_bc
        }

        if epoch % 10 == 0:
            data['g_params'] = generator.state_dict()
            data['d_params'] = discriminator.state_dict()

        # Save to checkpoint file
        pickle.dump(data, checkpoint_file)
        checkpoint_file.flush()

        # Print status
        print(f'Epoch: {epoch:4d} - Time: {time.time() - epoch_start_time:.2f} - MSE_r: {mse_pde:.2e} - L2_r: {L2_r:.2e} - L2_p: {L2_p:.2e} - L2_u: {L2_u:.2e} - L2_v: {L2_v:.2e}')
        
        epoch += 1
 
        # Termination conditions
        if torch.isnan(loss):
            checkpoint_file.close()
            break