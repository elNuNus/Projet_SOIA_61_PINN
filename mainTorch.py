import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(1, 20) if i == 0 else nn.Linear(20, 20) for i in range(8)])
        self.output_layer_mean = nn.Linear(20, 1)
        self.output_layer_var = nn.Linear(20, 1)

    def forward(self, x):
        y = x.unsqueeze(-1)
        for layer in self.hidden_layers:
            y = torch.tanh(layer(y))
        mean = self.output_layer_mean(y)
        std =  torch.abs(self.output_layer_var(y))
        return mean, std


def open_boundary(N, g_d_1=0, g_d_2=0):
    x = torch.cat([torch.ones(N // 2) * 1, torch.ones(N // 2) * -1], dim=0)
    u = torch.cat([torch.ones(N // 2) * g_d_1, torch.ones(N // 2) * g_d_2], dim=0)
    return x, u


def f(u, x, a, g):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return a * u_xx - g


def source(x, a):
    return torch.full_like(x, 2 * a)


def sol(x):
    return (x - 1) * (x + 1)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def mse_loss(input, target):
    return ( (input - target) ** 2)

def custom_loss(mean,log_var,target):
    return 0.5*((mean - target) ** 2)/torch.exp(log_var)+log_var

if __name__ == '__main__':
    N = 2000
    grid_x = torch.linspace(-1, 1, N, dtype=torch.float32, requires_grad=True)

    model = Network()
    model.apply(initialize_weights)

    # Boundary loss
    N_SAMPLE_POINTS_BND = 100
    a = 1
    x_bc, u_bc = open_boundary(N_SAMPLE_POINTS_BND)
    x_bc, u_bc = x_bc.float().unsqueeze(-1), u_bc.float().unsqueeze(-1)

    # Physics loss inside of domain
    N_SAMPLE_POINTS_INNER = 2000
    #x_ph = torch.rand(N_SAMPLE_POINTS_INNER, dtype=torch.float32) * 2 - 1
    x_ph = torch.rand(N_SAMPLE_POINTS_INNER, dtype=torch.float32) * (2-1e-5) - (1-1e-5)
    x_ph.requires_grad = True
    g = source(x_ph, a)

    ph_factor = 1.0
    #optimizer = optim.SGD(model.parameters(), lr=0.02)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ITERS = 5000
    loss_plot = []
    for step in range(ITERS + 1):
        optimizer.zero_grad()

        # Boundary loss
        u_pred_bc,s_bc = model(x_bc)
        loss_u = torch.nn.functional.mse_loss(u_pred_bc.squeeze(-1), u_bc)

        # Physics loss
        u_pred_ph,s_ph = model(x_ph.unsqueeze(-1))
        loss_ph = custom_loss(f(u_pred_ph, x_ph, a, g), s_ph,torch.zeros_like(g)).mean()

        # Total loss
        loss = loss_u + ph_factor * loss_ph
        loss.backward()
        optimizer.step()

        loss_plot.append(loss.item())
        if step < 11 or step % 100 == 0:
            print(f'Step {step}, loss: {loss.item()}')

    # Plot loss
    plt.figure()
    plt.plot(loss_plot, label='Loss')
    plt.legend()
    plt.show()

    # Plot final prediction vs. true solution
    with torch.no_grad():
        u_after,std = model(grid_x.unsqueeze(-1))
        print(np.exp(std.numpy()[:,0,0]))
    plt.figure()
    std = np.exp(std.numpy()[:,0,0])
    plt.plot(u_after.numpy()[:,0,0], label='NN')
    plt.plot(u_after.numpy()[:,0,0]-3*(1-np.sqrt(std)),'g--')
    plt.plot(u_after.numpy()[:,0,0]+3*(1-np.sqrt(std)),'g--')
    plt.plot(sol(grid_x).detach().numpy(), label='True')
    plt.legend()
    plt.show()
