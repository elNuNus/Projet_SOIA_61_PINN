import numpy as np
from phi.tf.flow import *
import matplotlib.pyplot as plt

rnd = math.choose_backend(1)

def network(x):
    y = math.stack([x], axis=-1)
    for i in range(8):
        y = tf.layers.dense(y, 20, activation=tf.math.tanh, name='layer%d' % i, reuse=tf.AUTO_REUSE)
    return tf.layers.dense(y, 1, activation=None, name='layer_out', reuse=tf.AUTO_REUSE)


def open_boundary(N, g_d_1=0,g_d_2=0):
    """Returns the boundary points and associated values."""
    x = math.concat([math.zeros([N // 2]) + 1, math.zeros([N // 2]) - 1], axis=0)
    u = math.concat([math.zeros([N // 2]) +g_d_1, math.zeros([N // 2]) +g_d_2], axis=0)
    return x, u


def f(u, x, a, g):
    """ Physics-based loss function """
    u_x = gradients(u, x)
    u_xx = gradients(u_x, x)
    return a*u_xx-g

def source(x, a):
    return 2*a

def sol(x):
    return (x-1)*(x+1)

if __name__ == '__main__':
    N = 2000
    grid_x = np.asarray(np.linspace(-1, 1, N),dtype=np.float32)
    print(grid_x.dtype)

    # in this case gives shape=(1, N, 1)
    grid_u = math.expand_dims(network(grid_x))

    print("Size of grid_u: " + format(grid_u.shape))
    session = Session(None)
    session.initialize_variables()
    u_init = session.run(grid_u)
    plt.figure()
    plt.plot(np.reshape(u_init,-1))
    plt.show()

    # Boundary loss
    N_SAMPLE_POINTS_BND = 100
    a=1
    x_bc, u_bc = open_boundary(N_SAMPLE_POINTS_BND)
    x_bc, u_bc = np.asarray(x_bc, dtype=np.float32), np.asarray(u_bc,dtype=np.float32)
    print(x_bc)
    print(u_bc)
    print(network(x_bc))
    loss_u = math.l2_loss(network(x_bc)[:, 0] - u_bc)

    # Physics loss inside of domain
    N_SAMPLE_POINTS_INNER = 1000
    x_ph = rnd.random_uniform([N_SAMPLE_POINTS_INNER], -1, 1)
    g = tf.convert_to_tensor(np.asarray(source(x_ph,1),dtype=np.float32))
    x_ph = tf.convert_to_tensor(x_ph)
    loss_ph = math.l2_loss(f(network(x_ph)[:, 0], x_ph,a,g))

    # Combine
    ph_factor = 1.
    loss = loss_u + ph_factor * loss_ph  # allows us to control the relative influence of loss_ph

    optim = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)

    session.initialize_variables()

    import time

    start = time.time()

    ITERS = 5000
    loss_plot = []
    for optim_step in range(ITERS + 1):
        _, loss_value = session.run([optim, loss])
        loss_plot.append(loss_value)
        if optim_step < 3 or optim_step % 1000 == 0:
            print('Step %d, loss: %f' % (optim_step, loss_value))
            # show_state(grid_u)

    end = time.time()
    print("Runtime {:.2f}s".format(end - start))

    u_after = session.run(grid_u)
    plt.figure()
    plt.plot(loss_plot,label='Loss')
    plt.legend()
    plt.figure()
    plt.plot(np.reshape(u_after, -1),label='NN')
    plt.plot(sol(grid_x),label='True')
    plt.legend()
    plt.show()
