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
M_inf = 2.5
v_inf = 0.0
u_inf = math.sqrt(gamma*p_inf/r_inf)*M_inf

# Geometry
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0

# Hyperparameters
num_ib = 500
num_r = 10000


''' Partial differential equations '''
def pde (state, coords, viscosity):
    # Coordinates
    x = coords[0]
    y = coords[1]

    # State
    r = state[:, [0]]
    p = state[:, [1]]
    U = state[:, [2]]
    phi = state[:,[3]]
    u = U*torch.cos(phi)
    v = U*torch.sin(phi)
    
    rE = p/(gamma - 1) +0.5*r*(u**2+v**2)

    u1 = r
    u2 = r*u
    u3 = r*v
    u4 = rE

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

    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1),create_graph=True)[0]
    u2_x = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2),create_graph=True)[0]
    u3_x = torch.autograd.grad(u3, x, grad_outputs=torch.ones_like(u3),create_graph=True)[0]
    u4_x = torch.autograd.grad(u4, x, grad_outputs=torch.ones_like(u4),create_graph=True)[0]

    u1_xx = torch.autograd.grad(u1_x, x, grad_outputs=torch.ones_like(u1_x),create_graph=True)[0]
    u2_xx = torch.autograd.grad(u2_x, x, grad_outputs=torch.ones_like(u2_x),create_graph=True)[0]
    u3_xx = torch.autograd.grad(u3_x, x, grad_outputs=torch.ones_like(u3_x),create_graph=True)[0]
    u4_xx = torch.autograd.grad(u4_x, x, grad_outputs=torch.ones_like(u4_x),create_graph=True)[0]

    u1_y = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1),create_graph=True)[0]
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2),create_graph=True)[0]
    u3_y = torch.autograd.grad(u3, y, grad_outputs=torch.ones_like(u3),create_graph=True)[0]
    u4_y = torch.autograd.grad(u4, y, grad_outputs=torch.ones_like(u4),create_graph=True)[0]

    u1_yy = torch.autograd.grad(u1_y, y, grad_outputs=torch.ones_like(u1_y),create_graph=True)[0]
    u2_yy = torch.autograd.grad(u2_y, y, grad_outputs=torch.ones_like(u2_y),create_graph=True)[0]
    u3_yy = torch.autograd.grad(u3_y, y, grad_outputs=torch.ones_like(u3_y),create_graph=True)[0]
    u4_yy = torch.autograd.grad(u4_y, y, grad_outputs=torch.ones_like(u4_y),create_graph=True)[0]

    r1 = f1_x + g1_y - viscosity * (u1_xx + u1_yy)
    r2 = f2_x + g2_y - viscosity * (u2_xx + u2_yy)
    r3 = f3_x + g3_y - viscosity * (u3_xx + u3_yy)
    r4 = f4_x + g4_y - viscosity * (u4_xx + u4_yy)

    return r1, r2, r3, r4



''' Boundary conditions'''
def collocation_points (device='cpu'):
    p1 = torch.tensor([0.5, 0.0], device=device).reshape(-1,2)
    p2 = torch.tensor([1.0, 0.25], device=device).reshape(-1,2)
    p3 = torch.tensor([2.0, 0.25], device=device).reshape(-1,2)

    # Left boundary
    x_l = torch.full((num_ib,1), x_min, device=device)
    y_l = torch.linspace(y_min, y_max, num_ib, device=device).reshape(-1,1)

    # Top boundary
    x_t = torch.linspace(x_min, x_max, num_ib, device=device).reshape(-1,1)
    y_t = torch.full((num_ib, 1), y_max, device=device)

    t_t = torch.full((num_ib,1), 0.0,device=device)

    # Bottom boundary
    x_b = torch.linspace(x_min,p1[0,0], num_ib+1, device=device)[:-1].reshape(-1,1)
    y_b = torch.full((num_ib,1), y_min, device=device)

    t_b = torch.full((num_ib,1), 0.0, device=device)

    # Wedge boundary
    t = torch.linspace(0,1,1000, device=device).reshape(-1,1)
    B = (1-t)**2 * p1 + 2*(1-t)*t*p2 + t**2 * p3
    dB = 2*(1-t)*(p2-p1) + 2*t*(p3-p2)

    # Wedge boundary
    x_w = B[:,[0]]
    y_w = B[:,[1]]

    t_w = torch.arctan(dB[:,[1]]/dB[:,[0]])

    # Residual points
    hammersley = skopt.sampler.Hammersly()
    x_r = torch.tensor(hammersley.generate([(x_min,x_max), (y_min, y_max)], num_r, random_state=0), device=device)

    def f (xs):
        result = torch.zeros_like(xs)

        for i in range(xs.shape[0]):
            t_ind = torch.argmin(torch.abs(B[:,0] - xs[i]))
            result[i] = B[t_ind,1]

        return result

    # Remove part below curve
    mapping = (x_r[:,1] > f(x_r[:,0])) | (x_r[:,0] < p1[0,0])
    x_r = x_r[mapping, :] 

    # Convert to tuples
    x_l = (x_l, y_l)
    x_t = (x_t, y_t)
    x_b = (x_b, y_b)
    x_w = (x_w, y_w)
    x_r = (x_r[:,[0]], x_r[:,[1]])

    return x_l, x_t, t_t, x_b, t_b, x_w, t_w, x_r


''' Reference data '''
def reference (device='cpu'):
    data = np.load('./curved/reference/reference.npz')
    x_test = torch.tensor(data['x_test'], device=device, dtype=torch.float64)
    y_test = torch.tensor(data['y_test'], device=device, dtype=torch.float64)
    u_test = torch.tensor(data['u_test'], device=device, dtype=torch.float64)

    x_test = x_test[~u_test[:,0].isnan(), :]
    y_test = y_test[~u_test[:,0].isnan(), :]
    u_test = u_test[~u_test[:,0].isnan(), :]

    # Convert to tuple
    x_test = (x_test, y_test)

    return x_test, u_test



''' Plot points '''
# import matplotlib.pyplot as plt

# x_l, x_t, t_t, x_b, t_b, x_w, t_w, x_r, x_o = collocation_points()

# plt.scatter(x_l[0], x_l[1], label='Left boundary',zorder=6)
# plt.scatter(x_o[0], x_o[1], label='Outer boundary',zorder=6)
# plt.scatter(x_t[0], x_t[1], label='Top boundary', zorder=6)
# plt.scatter(x_b[0], x_b[1], label='Bottom boundary',zorder=6)
# plt.scatter(x_w[0], x_w[1], label='Wedge boundary', zorder=5)
# plt.scatter(x_r[0], x_r[1], label='Residual points',zorder=4)
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
        # Force positive density and pressure, and magnitude of velocity
        res = self.map(torch.cat(a, 1))
        res[:,0] = torch.exp(res[:,0])
        res[:,1] = torch.exp(res[:,1])

        res[:,3] = torch.tanh(res[:,3]) * torch.pi / 2

        return res
    
# Discriminator
class Discriminator (torch.nn.Module):
    def __init__(self):
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
        res = self.map(torch.cat(a, 1))
        return res


def loss_slip (coords, thetas):
    state = generator(coords)
    disc = discriminator(coords)

    phi = state[:,[3]]
    loss = (disc[:,[8]] * (phi - thetas)).mean()
    mse = ((phi - thetas)**2).mean().item()
    
    return loss, mse

def loss_outlet (x_bc):
    u_pred = generator(x_bc)

    r = u_pred[:,[0]]
    p = u_pred[:,[1]]
    U = u_pred[:,[2]]

    c = torch.sqrt(gamma * p / r)
    M = U / c
    
    # Only subsonic flow
    mapping = M[:,0] < 1

    M = M[mapping,:]
    p = p[mapping,:]

    if M.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    else:
        return ((p - p_inf)**2).mean()


# Loss function for PDE
def loss_pde (coords):
    for x in coords: x.requires_grad = True
    state = generator(coords)
    r1, r2, r3, r4 = pde(state, coords, 5e-3)

    for x in coords: x.requires_grad = False
    disc = discriminator(coords)

    loss = (disc[:,[0]] * r1).mean() + \
           (disc[:,[1]] * r2).mean() + \
           (disc[:,[2]] * r3).mean() + \
           (disc[:,[3]] * r4).mean()
    
    mse = (r1**2).mean().item() + \
          (r2**2).mean().item() + \
          (r3**2).mean().item() + \
          (r4**2).mean().item()
        
    return loss, mse

def loss_inlet (x_bc):
    u_pred = generator(x_bc)
    disc = discriminator(x_bc)

    r = u_pred[:, [0]]
    p = u_pred[:, [1]]
    U = u_pred[:, [2]]
    phi = u_pred[:,[3]]
    u = U*torch.cos(phi)
    v = U*torch.sin(phi)

    
    loss = (disc[:,[4]] * (r-r_inf)).mean() + \
           (disc[:,[5]] * (p-p_inf)).mean() + \
           (disc[:,[6]] * (u-u_inf)).mean() + \
           (disc[:,[7]] * (v-v_inf)).mean()

    mse = ((r - r_inf)**2).mean().item() + \
          ((p - p_inf)**2).mean().item() + \
          ((u - u_inf)**2).mean().item() + \
          ((v - v_inf)**2).mean().item()
    
    return loss, mse

def evaluate ():
    global mse_bc, mse_pde, closs_r, closs_bc
    optimizer.zero_grad()                     

    loss_r, mse_r = loss_pde(x_r)        
    loss_b, mse_b = loss_slip(x_b, t_b)
    loss_l, mse_l = loss_inlet(x_l)
    loss_c, mse_c = loss_slip(x_w, t_w) 

    loss_bc = loss_l + loss_b + loss_c

    mse_bc = mse_l + mse_b + mse_c
    mse_pde = mse_r
    
    closs_r = loss_r.item()
    closs_bc = loss_bc.item()

    loss = loss_r + loss_bc

    return loss

if __name__ == '__main__':
    # Set up hardware 
    device = torch.device('cuda')
    torch.set_default_dtype(torch.float64)

    # Load collocation points and data
    x_l, x_t, t_t, x_b, t_b, x_w, t_w, x_r = collocation_points(device)
    x_test, u_test = reference(device)

    # Set up models and optimizer
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    solver = torch_cgd.solvers.GMRES(tol=1e-7, atol=1e-20)
    optimizer = torch_cgd.ACGD(generator.parameters(), discriminator.parameters(), lr=0.001, solver=solver, eps=1e-8, beta=0.99)

    # Prepare logging
    checkpoint_file = open(f'./curved/cpinn_phi_viscosity.p', 'ab')

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

        # State
        r = u_pred[:, 0]
        p = u_pred[:, 1]
        U = u_pred[:, 2]
        phi = u_pred[:,3]
        u = U*torch.cos(phi)
        v = U*torch.sin(phi)

        L2_r = torch.linalg.norm(r - u_test[:,0],2) / torch.linalg.norm(u_test[:,0],2)
        L2_p = torch.linalg.norm(p - u_test[:,1],2) / torch.linalg.norm(u_test[:,1],2)
        L2_u = torch.linalg.norm(u - u_test[:,2],2) / torch.linalg.norm(u_test[:,2],2)
        L2_v = torch.linalg.norm(v - u_test[:,3],2) / torch.linalg.norm(u_test[:,3],2)
        
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
    