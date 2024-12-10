import numpy as np
from phi.tf.flow import *
import matplotlib.pyplot as plt

rnd = math.choose_backend(1)


def network(x,y):
    y = math.stack([x,y], axis=-1)
    for i in range(8):
        y = tf.layers.dense(y, 20, activation=tf.math.softplus, name='layer%d' % i, reuse=tf.AUTO_REUSE)
    return tf.layers.dense(y, 1, activation=None, name='layer_out', reuse=tf.AUTO_REUSE)


def open_boundary(N):
    x_plus = rnd.random_uniform([N//2], -1, 1)
    y_plus = rnd.random_uniform([N//2], -1, 1)
    x = math.concat([math.zeros([N // 4]) + 1, math.zeros([N // 4]) - 1,x_plus], axis=0)
    y = math.concat([y_plus,math.zeros([N // 4]) + 1, math.zeros([N // 4]) - 1], axis=0)
    u = math.concat([math.zeros([N // 4]) , math.zeros([N // 4]) ,math.zeros([N // 4]), math.zeros([N // 4])], axis=0)
    return x, y,u

def neumann_boundary(N):
    x = rnd.random_uniform([N], -1, 1)
    y = math.concat([math.zeros([N // 2]) + 1, math.zeros([N // 2]) - 1], axis=0)
    u = math.concat([math.zeros([N // 2]), math.zeros([N // 2])], axis=0)
    return x, y, u

def loss_Neumann(x,y,u,u_theta):
    u_n = gradients(u_theta, x)
    return math.l2_loss(u_n - u)

def dirichlet_boundary(N):
    y = rnd.random_uniform([N], -1, 1)
    x = math.concat([math.zeros([N // 2]) + 1, math.zeros([N // 2]) - 1], axis=0)
    u = math.concat([math.zeros([N // 2]) +1, math.zeros([N // 2]) -1], axis=0)
    return x, y,u


def f(u, x, y,g,a=1):
    """ Physics-based loss function """
    u_x = gradients(u, x)
    u_y = gradients(u,y)
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y,y)
    return a*(u_xx+u_yy)-g

def source(x,y,a=1):
    N = len(x)
    g = np.zeros(N)
    # for i in range(N):
    #     g[i] = 2*a*(y[i]+1)*(y[i]-1)+2*a*(x[i]+1)*(x[i]-1)

    return np.asarray(g,dtype=np.float32)

def sol(x,y):
    N = len(x)
    sol = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            sol[i,j]= (x[i]-1)*(x[i]+1)*(y[j]+1)*(y[j]-1)
    return sol

def sol_cond(x,y,xmin,xmax,Vmin,Vmax):
    N = len(x)
    sol = np.zeros((N, N))
    for i in range(N):
        x_s = x[i]
        for j in range(N):
            sol[i, j] = Vmin + (x_s-xmin)*(Vmax-Vmin)/(xmax-xmin)
    return sol

if __name__ == '__main__':
    N = 256
    grid_x = np.asarray(np.linspace(-1, 1, N),dtype=np.float32)
    grid_y = np.asarray(np.linspace(-1, 1, N),dtype=np.float32)
    sol_p = sol_cond(grid_x,grid_y,-1,1,-1,1)
    grids_xy = np.meshgrid(grid_x, grid_y, indexing='ij')


    grid_x, grid_y = [tf.convert_to_tensor(t, tf.float32) for t in grids_xy]
    # in this case gives shape=(1, N, N,1)
    grid_u = math.expand_dims(network(grid_x,grid_y))

    print("Size of grid_u: " + format(grid_u.shape))
    session = Session(None)
    session.initialize_variables()
    u_init = session.run(grid_u)
    #print(u_init[0,:,:,0])
    plt.figure()
    plt.imshow(u_init[0,:,:,0])
    plt.show()

    # Neumann Boundary loss
    N_SAMPLE_POINTS_BND = 500
    x_bc_n_p, y_bc_n_p,u_bc_n_p = neumann_boundary(N_SAMPLE_POINTS_BND)
    x_bc_n,y_bc_n, u_bc_n = tf.convert_to_tensor(x_bc_n_p), tf.convert_to_tensor(y_bc_n_p),tf.convert_to_tensor(u_bc_n_p)

    loss_n = loss_Neumann(x_bc_n,y_bc_n,u_bc_n,network(x_bc_n,y_bc_n)[:, 0])

    # # Boundary loss
    N_SAMPLE_POINTS_BND = 10000
    a=1
    x_bc, y_bc,u_bc = dirichlet_boundary(N_SAMPLE_POINTS_BND)
    x_bc,y_bc, u_bc = np.asarray(x_bc, dtype=np.float32), np.asarray(y_bc,dtype=np.float32),np.asarray(u_bc,dtype=np.float32)

    loss_u = math.l2_loss(network(x_bc,y_bc)[:, 0] - u_bc)


    # add Neumann
    # # Physics loss inside of domain
    N_SAMPLE_POINTS_INNER = 8000
    x_ph_p = rnd.random_uniform([N_SAMPLE_POINTS_INNER], -1+1e-6, 1-1e-6)
    y_ph_p = rnd.random_uniform([N_SAMPLE_POINTS_INNER], -1+1e-6, 1-1e-6)
    g = source(x_ph_p,y_ph_p)
    x_ph = tf.convert_to_tensor(x_ph_p)
    y_ph = tf.convert_to_tensor(y_ph_p)
    g = tf.convert_to_tensor(g)
    loss_ph = math.l2_loss(f(network(x_ph,y_ph)[:, 0], x_ph,y_ph,g))

    # # Combine
    bc_factor = 1
    ph_factor = 1
    loss = bc_factor*(3*loss_u+1/10*loss_n) + ph_factor * loss_ph  # allows us to control the relative influence of loss_ph

    optim = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)

    session.initialize_variables()

    import time

    start = time.time()

    ITERS = 10000
    loss_plot = []
    for optim_step in range(ITERS + 1):
        _, loss_value = session.run([optim, loss])
        loss_plot.append(loss_value)
        if optim_step < 10 or optim_step % 1000 == 0:
            print('Step %d, loss: %f' % (optim_step, loss_value))
            # show_state(grid_u)

    end = time.time()
    print("Runtime {:.2f}s".format(end - start))

    u_after = session.run(grid_u)
    plt.figure()
    plt.plot(loss_plot,label='Loss')
    plt.legend()
    plt.figure()
    plt.imshow(u_after[0,:,:,0])
    plt.colorbar()
    plt.figure()
    plt.imshow(sol_p)
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(u_after[0,:,:,0]-sol_p))
    plt.colorbar()

    fig, ax = plt.subplots()

    plt.plot(x_bc, y_bc, '*', color='red', markersize=5, label='Boundary (Dir) collocation points= 500')
    plt.plot(x_bc_n_p, y_bc_n_p, '*', color='green', markersize=5, label='Boundary (Neu) collocation points= 500')

    plt.plot(x_ph_p, y_ph_p, 'o', markersize=0.5, label='PDE collocation points = 10000')

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Collocation points')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('scaled')
    plt.show()
    plt.show()
