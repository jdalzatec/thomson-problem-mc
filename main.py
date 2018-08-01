import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import click

# return a random configuration of n particles over a unitary sphere
def random_configuration(n):
    points = numpy.random.normal(size=(n, 3))
    norms = numpy.linalg.norm(points, axis=1)
    points = numpy.array([points[i] / norms[i] for i in range(n)])
    norms = numpy.linalg.norm(points, axis=1)
    assert numpy.allclose(norms, numpy.ones_like(norms))
    return points

def plot_configuration(positions, show=False, out="configuration.pdf"):
    x, y, z = positions.T
    fig = pyplot.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    u = numpy.linspace(0, 2 * numpy.pi, 200)
    v = numpy.linspace(0, numpy.pi, 200)
    lx = 1.0 * numpy.outer(numpy.cos(u), numpy.sin(v))
    ly = 1.0 * numpy.outer(numpy.sin(u), numpy.sin(v))
    lz = 1.0 * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

    # Plot the surface
    ax.plot_surface(lx, ly, lz, color='b', alpha=0.1, lw=0, rstride=4, cstride=4,)

    u = numpy.linspace(0, 2 * numpy.pi, 20)
    v = numpy.linspace(0, numpy.pi, 20)
    lx = 1.0 * numpy.outer(numpy.cos(u), numpy.sin(v))
    ly = 1.0 * numpy.outer(numpy.sin(u), numpy.sin(v))
    lz = 1.0 * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))
    ax.plot_wireframe(lx, ly, lz, color='b', alpha=0.1, lw=0.5)

    # Plot the surface lines
    ax.scatter(x, y, z, s=10, color="crimson", edgecolor="black", lw=0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_aspect("equal")
    ax.axis('off')
    pyplot.tight_layout()
    pyplot.savefig("small_trial.pdf")
    if show:
        pyplot.show()
    pyplot.close()
    

@click.command()
@click.option("-n", default=50)
def main(n):
    some = random_configuration(n)
    plot_configuration(some)




if __name__ == '__main__':
    main()