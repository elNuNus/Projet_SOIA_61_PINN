import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# from bayes_opt import BayesianOptimization


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(1, 20) if i == 0 else nn.Linear(20, 20) for i in range(8)])
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x):
        y = x.unsqueeze(-1)
        for layer in self.hidden_layers:
            y = torch.tanh(layer(y))
        out = self.output_layer(y)
        return out


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


def train(x_bc,u_bc,x_ph,N_epoch=500,w_bc=1,w_ph=1):

    model = Network()
    model.apply(initialize_weights)


    g = source(x_ph, a)

    # optimizer = optim.SGD(model.parameters(), lr=0.02)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ITERS = N_epoch
    loss_plot = []
    for step in range(ITERS + 1):
        optimizer.zero_grad()

        # Boundary loss
        u_pred_bc = model(x_bc)
        loss_u = torch.nn.functional.mse_loss(u_pred_bc.squeeze(-1), u_bc)

        # Physics loss
        u_pred_ph = model(x_ph.unsqueeze(-1))
        #loss_ph = mse_loss(f(u_pred_ph, x_ph, a, g), torch.zeros_like(g)).mean()
        loss = torch.nn.MSELoss()
        loss_ph = loss(f(u_pred_ph, x_ph, a, g),torch.zeros_like(g))
        # Total loss
        loss = w_bc*loss_u + w_ph * loss_ph
        loss.backward()
        optimizer.step()

        loss_plot.append(loss.item())
        if step < 11 or step % 100 == 0:
             print(f'Step {step}, loss: {loss.item()}')

    N = 2000
    grid_x = torch.linspace(-1, 1, N, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        u_after = model(grid_x.unsqueeze(-1))

    #print(u_after.numpy()[:, 0, 0])
    #print(sol(grid_x).detach().numpy())
    mse = np.linalg.norm(u_after.numpy()[:, 0, 0]-sol(grid_x).detach().numpy(),2)
    # print(mse)

    # plt.figure()
    # plt.plot(loss_plot, label='Loss')
    # plt.legend()
    # plt.show()

    with torch.no_grad():
        u_after =  model(grid_x.unsqueeze(-1))
        #print(np.exp(std.numpy()[:, 0, 0]))
    plt.figure()
    plt.plot(u_after.numpy()[:, 0, 0], label='NN')
    plt.plot(sol(grid_x).detach().numpy(), label='True')
    plt.legend()
    plt.show()

    return model



if __name__ == '__main__':
    N_SAMPLE_POINTS_BND = 100
    a = 1
    x_bc, u_bc = open_boundary(N_SAMPLE_POINTS_BND)
    x_bc, u_bc = x_bc.float().unsqueeze(-1), u_bc.float().unsqueeze(-1)

    # Physics loss inside of domain
    N_SAMPLE_POINTS_INNER = 2000
    # x_ph = torch.rand(N_SAMPLE_POINTS_INNER, dtype=torch.float32) * 2 - 1
    x_ph = torch.rand(N_SAMPLE_POINTS_INNER, dtype=torch.float32) * (2 - 1e-5) - (1 - 1e-5)
    x_ph.requires_grad = True



    train(x_bc,u_bc,x_ph,1000,1,1)

    # # Plot collocation points
    # plt.figure()
    # plt.scatter(x_bc, u_bc, label='Boundary')
    # plt.scatter(x_ph, torch.zeros_like(x_ph), label='Physics')
    # plt.legend()
    # plt.show()

