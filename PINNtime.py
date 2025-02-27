import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(2, 20) if i == 0 else nn.Linear(20, 20) for i in range(8)])
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x, t):
        y = torch.stack([x,t],dim=-1)
        # print("y shape :",y.shape)
        for layer in self.hidden_layers:
            y = torch.tanh(layer(y))
        out = self.output_layer(y)
        # print("out shape :",out.shape)
        return out

def open_boundary(N, g_d_1=1, g_d_2=0):
    t = torch.rand(N, dtype=torch.float32)
    x = torch.cat([torch.ones(N // 2) * 1, torch.ones(N // 2) * -1], dim=0)

    u = torch.cat([torch.ones(N // 2) * g_d_1, torch.ones(N // 2) * g_d_2], dim=0)
    return x,t, u

def init_cond(N):
    x = torch.rand(N, dtype=torch.float32)*2-1
    t = torch.zeros_like(x)
    u = - torch.sin(np.pi * x)
    return x, t, u

def f(u, x, t, a):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return u_t-a * u_xx

N = 128
grids_xt = np.meshgrid(np.linspace(-1, 1, N), np.linspace(0, 1, 33), indexing='ij')
grid_x, grid_t = [torch.tensor(t,dtype=torch.float32) for t in grids_xt]
print("grid_x shape :",grid_x.shape)
print("grid_t shape :",grid_t.shape)
print("grids_xt shape :",grids_xt[0].shape)

model = Network()

u_pred_bc = model(grid_x,grid_t)


N_bc = 100

x_bc, t_bc, u_bc = [torch.stack([v_t0, v_x], dim=0) for v_t0, v_x in zip(init_cond(N_bc), open_boundary(N_bc,0,0))]
#x_bc, t_bc, u_bc = np.asarray(x_bc,dtype=np.float32), np.asarray(t_bc,dtype=np.float32) ,np.asarray(u_bc,dtype=np.float32)

print("u_bc shape:",u_bc.shape)

N_in = 1000
x_ph, t_ph = torch.rand(N_in)*2-1,torch.rand(N_in)
x_ph.requires_grad = True
t_ph.requires_grad = True

optimizer = optim.SGD(model.parameters(), lr=0.1)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
ITERS = 5000
loss_plot = []
for step in range(ITERS + 1):
    optimizer.zero_grad()

    # Boundary loss
    u_pred_bc = model(x_bc,t_bc)
    # print(u_pred_bc.squeeze(-1))
    # print(u_bc)
    loss_u = torch.nn.functional.mse_loss(u_pred_bc.squeeze(-1), u_bc)


    # Physics loss
    u_pred_ph = model(x_ph,t_ph)
    # print("u_pred_ph shape :",u_pred_ph.shape)
    #loss_ph = mse_loss(f(u_pred_ph, x_ph, a, g), torch.zeros_like(g)).mean()
    loss_ph = torch.nn.functional.mse_loss(f(u_pred_ph,x_ph,t_ph,0.1),torch.zeros(N_in))
    # Total loss
    loss = loss_u +  loss_ph
    loss.backward()
    optimizer.step()

    loss_plot.append(loss.item())
    if step < 11 or step % 100 == 0:
         print(f'Step {step}, loss: {loss.item()}')
#%%
# Plot loss
plt.plot(loss_plot)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# Plot 2D solution
u_pred = model(grid_x,grid_t).detach().numpy().reshape(N, 33)
print("u_pred shape :",u_pred.shape)
plt.imshow(u_pred, aspect='auto', extent=(-1, 1, 0, 1))
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('Predicted solution')
plt.show()

plt.show()
#%%
# Plot the solutions in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, T = np.meshgrid(np.linspace(-1, 1, N), np.linspace(0, 1, 33))
ax.plot_surface(X, T, u_pred.T, cmap='viridis')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u")
plt.show()