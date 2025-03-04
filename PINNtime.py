import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

n_layers = 10
n_neurons = 100

nu = 0.1
v = 1

class Network(nn.Module):
    def __init__(self, neurons=20, layers=8):
        super(Network, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(2, neurons) if i == 0 else nn.Linear(neurons, neurons) for i in range(layers)])
        self.output_layer = nn.Linear(neurons, 1)

    def forward(self, x,t):
        # Stacking the tensors here allow for easier manipulation of the time and space variables, especially for derivatives
        y = torch.stack([x,t],dim=-1)
        # print("y shape : ", y.shape)
        for layer in self.hidden_layers:
            y = torch.tanh(layer(y))
        out = self.output_layer(y)
        # print("out shape : ", out.shape)
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

def f(u, x, t):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return u_t - nu*u_xx #+ v*u_x

N_x = 128
N_t = 150
grids_xt = np.meshgrid(np.linspace(-1, 1, N_x), np.linspace(0, 1, N_t), indexing="ij")
grid_x, grid_t = [torch.tensor(t,dtype=torch.float32) for t in grids_xt]
print("grid_x shape :",grid_x.shape)
print("grid_t shape :",grid_t.shape)
print("grids_xt shape :",grids_xt[0].shape)

model = Network(neurons=n_neurons, layers=n_layers)

u_pred_bc = model(grid_x,grid_t)


N_bc = 100

x_bc, t_bc, u_bc = [torch.stack([v_t0, v_x], dim=0) for v_t0, v_x in zip(init_cond(N_bc), open_boundary(N_bc,0,0))]
#x_bc, t_bc, u_bc = np.asarray(x_bc,dtype=np.float32), np.asarray(t_bc,dtype=np.float32) ,np.asarray(u_bc,dtype=np.float32)

# print("u_bc shape:",u_bc.shape)

N_in = 1000
x_ph, t_ph = torch.rand(N_in)*2-1,torch.rand(N_in)
x_ph.requires_grad = True
t_ph.requires_grad = True

# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
ITERS = 2000
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
    loss_ph = torch.nn.functional.mse_loss(f(u_pred_ph,x_ph,t_ph),torch.zeros(N_in))
    # Total loss
    loss = loss_u + loss_ph
    loss.backward()
    optimizer.step()

    loss_plot.append(loss.item())
    if step < 11 or step % 100 == 0:
         print(f"Step {step}, loss: {loss.item()}")

#%%
# Exact solution for the 1D heat equation
U_ex = -np.sin(np.pi * grids_xt[0]) * np.exp(-np.pi ** 2 * grids_xt[1] * nu)

# Plot loss
plt.plot(loss_plot)
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Training loss of the 1D convection-diffusion equation \n Architecture : {n_layers} layers, {n_neurons} neurons/layer \n Best loss : {np.min(loss_plot)}")
plt.show()

# Plot 2D solution
u_pred = model(grid_x,grid_t).detach().numpy().reshape(N_x, N_t)
print("u_pred shape :",u_pred.shape)
plt.imshow(u_pred, aspect="auto", extent=(0, 1, -1, 1), cmap="plasma")
plt.colorbar()
plt.xlabel("t")
plt.ylabel("x")
# plt.title("Ground truth for the 1D heat equation")
plt.title(f"Solution of the 1D heat equation \n Architecture : {n_layers} layers, {n_neurons} neurons/layer")
# plt.title(f"Solution of the 1D convection-diffusion equation \n Architecture : {n_layers} layers, {n_neurons} neurons/layer")
plt.show()

u_pred = u_pred.T
#%%
# MSE computation
X, T = np.meshgrid(np.linspace(-1, 1, N_x), np.linspace(0, 1, N_t))
u_true = -np.sin(np.pi * X) * np.exp(-np.pi ** 2 * T)
mse = np.mean((u_true - u_pred) ** 2)
print("\nMSE: ", mse)

# Plot error map between the two
plt.figure()
plt.imshow(np.abs(u_pred.T - U_ex), aspect='auto', cmap="viridis", extent=(0, 1, -1, 1))
plt.xlabel("t")
plt.ylabel("x")
plt.title(f"Error map between the prediction and the ground truth \n Architecture : {n_layers} layers, {n_neurons} neurons/layer")
plt.colorbar()
plt.show()
#%%
# Plot the solutions in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, T, u_pred, cmap="plasma")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u")
plt.title(f"Solution of the 1D convection-diffusion equation \n Architecture : {n_layers} layers, {n_neurons} neurons/layer")
plt.show()
#%% Plot the prediction and ground truth at t=.25
t_obs = 1
u_pred = u_pred.T
u_true = u_true.T
plt.plot(X[0], u_pred[:, t_obs], label="Prediction", color="green")
plt.plot(X[0], u_true[:, t_obs], label="Ground truth", color="blue", linestyle="--")
plt.legend()
plt.title("Prediction and ground truth at t=.25")
plt.show()