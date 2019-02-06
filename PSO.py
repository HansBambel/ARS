import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2


def rastrigin(x, y):
    return 10*2 * np.sum(x**2.0 - 10*np.cos(2*np.pi*x) + y**2.0 - 10*np.cos(2*np.pi*y))


def try_rast(plot_3d=False):
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


def try_rosen(plot_3d=False):
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
    fig = plt.figure()
    ### Init Particles
    if benchmark == 'rosenbrock':
        partPos = np.array([np.random.uniform(-2, 2, size=particles),
                           np.random.uniform(-1, 3, size=particles)]).T
        partVelocity = np.array([np.random.uniform(-4, 4, size=particles),
                                np.random.uniform(-4, 4, size=particles)]).T
        benchmarkFunction = lambda x, y: rosenbrock(x, y)
    elif benchmark == 'rastrigin':
        partPos = np.array([np.random.uniform(-5, 5, size=particles),
                           np.random.uniform(-5, 5, size=particles)]).T
        partVelocity = np.array([np.random.uniform(-10, 10, size=particles),
                                np.random.uniform(-10, 10, size=particles)]).T
        benchmarkFunction = lambda x, y: rastrigin(x, y)
    else:
        partPos = np.array([np.random.uniform(-2, 2, size=particles),
                           np.random.uniform(-1, 3, size=particles)]).T
        partVelocity = np.array([np.random.uniform(-4, 4, size=particles),
                                np.random.uniform(-4, 4, size=particles)]).T
        benchmarkFunction = lambda x, y: rosenbrock(x, y)

    partBestPos = np.copy(partPos)
    globBestPos = partBestPos[0]
    globBest = benchmarkFunction(globBestPos[0], globBestPos[1])
    for p in partBestPos:
        tmp = benchmarkFunction(p[0], p[1])
        if tmp < globBest:
            globBest = tmp
            globBestPos = p

    ### Let the particles do its thang
    maxIterations = 100
    for iter in range(maxIterations):
        for i, p in enumerate(partPos):
            for d in range(len(partPos[i])):
                rP = np.random.randint(0, 2)
                rG = np.random.randint(0, 2)
                # Update velocity
                partVelocity[i, d] = a*partVelocity[i, d] + \
                                     b*rP*(partBestPos[i, d] - p[d]) + \
                                     c*rG*(globBestPos[d] - p[d])
            # new Particle Position
            partPos += partVelocity
            if benchmarkFunction(p[0], p[1]) < benchmarkFunction(partBestPos[i, 0], partBestPos[i, 1]):
                partBestPos = p
                if benchmarkFunction(partBestPos[0], partBestPos[1]) < globBest:
                    globBest = benchmarkFunction(partBestPos[0], partBestPos[1])
                    globBestPos = p

        # Plot the particles
        plt.scatter(partPos[:, 0], partPos[:, 1])
        plt.draw()
        plt.pause(0.001)


particleSwarm()
