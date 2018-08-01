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

def plot_configuration(positions, charge_center=True, show=False, out="configuration.pdf"):
    x, y, z = positions.T
    fig = pyplot.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    u = numpy.linspace(0, 2.0 * numpy.pi, 200)
    v = numpy.linspace(0, numpy.pi, 200)
    lx = 1.0 * numpy.outer(numpy.cos(u), numpy.sin(v))
    ly = 1.0 * numpy.outer(numpy.sin(u), numpy.sin(v))
    lz = 1.0 * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

    # Plot the surface
    ax.plot_surface(lx, ly, lz, color='b', alpha=0.1, lw=0, rstride=4, cstride=4,)

    u = numpy.linspace(0, 2.0 * numpy.pi, 20)
    v = numpy.linspace(0, numpy.pi, 20)
    lx = 1.0 * numpy.outer(numpy.cos(u), numpy.sin(v))
    ly = 1.0 * numpy.outer(numpy.sin(u), numpy.sin(v))
    lz = 1.0 * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))
    ax.plot_wireframe(lx, ly, lz, color='b', alpha=0.1, lw=0.5)

    # Plot the surface lines
    ax.scatter(x, y, z, s=10, color="crimson", edgecolor="black", lw=0.5)

    if charge_center:
        xm, ym, zm = numpy.mean(positions, axis=0)
        ax.scatter(xm, ym, zm, s=10, color="black")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_aspect("equal")
    ax.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(out)
    if show:
        pyplot.show()
    pyplot.close()


def local_potential_energy(index, positions):
    assert (index < len(positions))
    norms = numpy.linalg.norm(positions - positions[index], axis=1)
    norms = norms[norms != 0.0]
    return numpy.sum(1.0 / norms)


def potential_energy(positions):
    energy = 0.0
    for i in range(len(positions)):
        energy += local_potential_energy(i, positions)
    return 0.5 * energy


def new_position_in_vicinity(position, sigma=0.1):
    new_position = position + numpy.random.normal(loc=0.0, scale=sigma, size=3)
    new_position /= numpy.linalg.norm(new_position)
    return new_position


def metropolis(positions, sigma=0.01, T=0.0001):
    for _ in range(len(positions)):
        randInt = numpy.random.randint(0, len(positions))
        oldPosition = positions[randInt]
        oldEnergy = local_potential_energy(randInt, positions)
        positions[randInt] = new_position_in_vicinity(positions[randInt], sigma)
        newEnergy = local_potential_energy(randInt, positions)
        deltaE = newEnergy - oldEnergy
        if deltaE <= 0:
            pass
        else:
            if numpy.random.uniform() <= numpy.exp(-deltaE / T):
                pass
            else:
                positions[randInt] = oldPosition



@click.command()
@click.option("-n", default=5)
def main(n):
    positions = random_configuration(n)
    plot_configuration(positions)
    E = local_potential_energy(0, positions)
    for _ in range(10000):
        metropolis(positions)
        print(potential_energy(positions))
    plot_configuration(positions, charge_center=True, show=False, out="new.pdf")



if __name__ == '__main__':
    main()