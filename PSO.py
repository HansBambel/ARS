import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rosenbrock(x, y):
    return (0 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rastrigin(x, y):
    return 10 * 2 * np.sum(x ** 2.0 - 10 * np.cos(2 * np.pi * x) + y ** 2.0 - 10 * np.cos(2 * np.pi * y))


def try_rast(plot_3d=False):
    X_rast = np.arange(-5, 5, 0.1)
    Y_rast = np.arange(-5, 5, 0.1)
    X_rast, Y_rast = np.meshgrid(X_rast, Y_rast)
    Z_rast = np.zeros(X_rast.shape)

    for i in np.arange(X_rast.shape[0]):
        for j in np.arange(X_rast.shape[1]):
            Z_rast[i, j] = rastrigin(X_rast[i, j], Y_rast[i, j])

    if plot_3d:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_rast, Y_rast, Z_rast, cmap=plt.cm.viridis)
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
            Z_rosen[i, j] = rosenbrock(X_rosen[i, j], Y_rosen[i, j])

    if plot_3d:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_rosen, Y_rosen, Z_rosen, cmap=plt.cm.viridis)
    else:
        plt.contourf(X_rosen, Y_rosen, Z_rosen, 100)
        plt.colorbar()
    plt.show()


def particleSwarm(particles=100, iterations=100, a=0.9, b=2, c=2, benchmark='rosenbrock', plotting=False):
    ### Init Particles
    if benchmark == 'rastrigin':
        xBounds = [-5, 5]
        yBounds = [-5, 5]
        benchmarkFunction = lambda x, y: rastrigin(x, y)
    else:
        xBounds = [-2, 2]
        yBounds = [-1, 3]
        benchmarkFunction = lambda x, y: rosenbrock(x, y)

    partPos = np.array([np.random.uniform(xBounds[0], xBounds[1], size=particles),
                        np.random.uniform(yBounds[0], yBounds[1], size=particles)]).T
    partVelocity = np.array([np.random.uniform(xBounds[0], xBounds[1], size=particles),
                             np.random.uniform(yBounds[0], yBounds[1], size=particles)]).T
    if plotting:
        X = np.arange(xBounds[0], xBounds[1], 0.1)
        Y = np.arange(yBounds[0], yBounds[1], 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)
        for i in np.arange(X.shape[0]):
            for j in np.arange(X.shape[1]):
                Z[i, j] = benchmarkFunction(X[i, j], Y[i, j])

    partBestPos = np.copy(partPos)
    partBestVal = np.zeros(len(partPos))
    globBestPos = partBestPos[0]
    globBest = benchmarkFunction(globBestPos[0], globBestPos[1])
    for i, p in enumerate(partBestPos):
        partBestVal[i] = benchmarkFunction(p[0], p[1])
        if partBestVal[i] < globBest:
            globBest = partBestVal[i]
            globBestPos = p


    ########################
    ### Let the particles do its thang
    for iter in range(iterations):
        for i, p in enumerate(partPos):
            rP = np.random.randint(0, 2)
            rG = np.random.randint(0, 2)
            # Update velocity
            partVelocity[i] = a * partVelocity[i] + \
                              b * rP * (partBestPos[i] - p) + \
                              c * rG * (globBestPos - p)
            # print(partVelocity[i])
            # new Particle Position (keep them in bounds)
            partPos[i] += partVelocity[i]
            partPos[i, 0] = min(max(partPos[i, 0], xBounds[0]), xBounds[1])
            partPos[i, 1] = min(max(partPos[i, 1], yBounds[0]), yBounds[1])
            partVal = benchmarkFunction(partPos[i, 0], partPos[i, 1])
            # print(f'Velocity before: {partPos[i] - partVelocity[i]} after: {partPos[i]}')
            if partVal <= partBestVal[i]:
                partBestPos[i] = partPos[i]
                partBestVal[i] = partVal
                if partBestVal[i] <= globBest:
                    globBestPos = partBestPos[i]
                    globBest = partBestVal[i]

        # Plot the particles
        if plotting:
            plot_particles(iter, iterations, X, Y, Z, partPos, globBestPos, xBounds, yBounds)
    if plotting:
        plot_particles(iter, iterations, X, Y, Z, partPos, globBestPos, xBounds, yBounds)

    return globBestPos, globBest


def plot_particles(iter, iterations, X, Y, Z, partPos, globBestPos, xBounds, yBounds):
    plt.clf()
    plt.title(f'Iteration {iter}/{iterations}')
    plt.contourf(X, Y, Z, 50, cmap='plasma')
    plt.plot(partPos[:, 0], partPos[:, 1], 'o', c='y', markersize=3)
    plt.plot(globBestPos[0], globBestPos[1], 'o', c='r')
    plt.xlim(xBounds[0], xBounds[1])
    plt.ylim(yBounds[0], yBounds[1])
    plt.draw()
    plt.pause(0.001)


benchmark = ['rosenbrock', 'rastrigin']
a, b, c = 0.5, 1, 1
bestPos, best = particleSwarm(particles=100, iterations=75, a=a, b=b, c=c, benchmark=benchmark[0], plotting=False)
print(f'Best Value: {np.round(best)} at Position X: {np.round(bestPos[0])} Y: {np.round(bestPos[1])}')
