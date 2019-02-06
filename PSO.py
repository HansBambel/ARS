import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

def rastrigin(x, y):
    return 10*2 * np.sum(x**2.0 - 10*np.cos(2*np.pi*x) + y**2.0 - 10*np.cos(2*np.pi*y))

def test_rast(plot_3d=False):
    X_rast = np.arange(-5, 5, 0.1)
    Y_rast = np.arange(-5, 5, 0.1)
    X_rast, Y_rast = np.meshgrid(X_rast, Y_rast)
    Z_rast = np.zeros(X_rast.shape)

    for i in np.arange(X_rast.shape[0]):
        for j in np.arange(X_rast.shape[1]):
            Z_rast[i,j] = rastrigin(X_rast[i,j],Y_rast[i,j])

    if plot_3d:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_rast,Y_rast,Z_rast, cmap=plt.cm.viridis)
    else:
        plt.contourf(X_rast, Y_rast, Z_rast)
        plt.colorbar()
    plt.show()

def test_rosen(plot_3d=False):
    X_rosen = np.arange(-2, 2, 0.1)
    Y_rosen = np.arange(-1, 3, 0.1)
    X_rosen, Y_rosen = np.meshgrid(X_rosen, Y_rosen)
    Z_rosen = np.zeros(X_rosen.shape)

    for i in np.arange(X_rosen.shape[0]):
        for j in np.arange(X_rosen.shape[1]):
            Z_rosen[i,j] = rosenbrock(X_rosen[i,j],Y_rosen[i,j])

    if plot_3d:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_rosen,Y_rosen,Z_rosen, cmap=plt.cm.viridis)
    else:
        plt.contourf(X_rosen, Y_rosen, Z_rosen, 100)
        plt.colorbar()
    plt.show()


def particleSwarm(particles=100, a=0.9, b=2, c=2, benchmark='rosenbrock'):
    if benchmark == 'rosenbrock':
        partPos = np.array(np.random.uniform(-2, 2, size=particles),
                           np.random.uniform(-1, 3, size=particles))
    elif benchmark == 'rastrigin':
        partPos = np.array(np.random.uniform(-5, 5, size=particles),
                           np.random.uniform(-5, 5, size=particles))
    else:
        partPos = np.array(np.random.uniform(-2, 2, size=particles),
                           np.random.uniform(-1, 3, size=particles))

    partBestPos = np.copy(partPos)



test_rosen(True)