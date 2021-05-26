import click
import numpy
import pandas
from matplotlib import pyplot
from tqdm import tqdm
from utils import plot_particle_distribution


def random_configuration(n):
    """Returns a random configuration of n particles over a unitary sphere.

    Arguments:
        n [int]: The number of particles.

    Returns:
        numpy.ndarray: Array with the particles distribution.
    """

    points = numpy.random.normal(size=(n, 3))
    norms = numpy.linalg.norm(points, axis=1)
    points = numpy.array([points[i] / norms[i] for i in range(n)])
    norms = numpy.linalg.norm(points, axis=1)
    assert numpy.allclose(norms, numpy.ones_like(norms))
    return points


def local_potential_energy(index, particles_distribution):
    """Computes the local potential energy for the particle in the position `index`.

    Arguments:
        index [int]: The particle index in the particles distribution.
        particles_distribution [numpy.ndarray]: The particles distribution.

    Returns:
        float: The local potential energy for the particle in the position `index`.
    """

    assert index < len(particles_distribution)
    norms = numpy.linalg.norm(
        particles_distribution - particles_distribution[index], axis=1
    )
    norms = norms[norms != 0.0]
    return numpy.sum(1.0 / norms)


def total_potential_energy(particles_distribution):
    """Computes the total potential energy.

    Arguments:
        particles_distribution [numpy.ndarray]: The particles distribution.

    Returns:
        float: The total potential energy for the particle in the position `index`.
    """

    energy = 0.0
    for i in range(len(particles_distribution)):
        energy += local_potential_energy(i, particles_distribution)
    return 0.5 * energy


def new_position_in_vicinity(position, sigma):
    """Returns a new position based on a `sigma` parameter. The lower `sigma`, the near
    the new point for the original position.

    Arguments:
        position [numpy.3darray]: The original position to be used to create a new one.
        sigma [float]: Displacement parameter.

    Returns:
        [numpy.3darray]: The new position.
    """

    new_position = position + sigma * numpy.random.normal(size=3)
    new_position /= numpy.linalg.norm(new_position)
    return new_position


def metropolis(index, particles_distribution, sigma, T):
    """Applies metropolis algorithm to generate a new state (new position) for the
    particle in the position `index` based on the temperature (T) and the parameter
    of displacement `sigma`. If the energy does not decrease when trying to create a new
    state, it guarantees that the original state is not altered.

    Arguments:
        index [int]: The index for the particle that we want to move.
        particles_distribution [numpy.ndarray]: The particles distribution.
        sigma [float]: Displacement parameter.
        T [float]: Temperature.

    Returns:
        bool: A boolean that indicates if it was created a new state or not.

    """

    assert T > 0
    old_position = particles_distribution[index].copy()
    old_energy = local_potential_energy(index, particles_distribution)

    particles_distribution[index] = new_position_in_vicinity(
        particles_distribution[index], sigma
    )
    new_energy = local_potential_energy(index, particles_distribution)

    delta_energy = new_energy - old_energy

    if numpy.random.uniform() > numpy.exp(-delta_energy / T):
        particles_distribution[index] = old_position
        return False

    return True


def monte_carlo_step(particles_distribution, sigma, T):
    """Applies N times the metropolis algorith on random positions, where N is the
    number of particles in the system.

    Arguments:
        particles_distribution [numpy.ndarray]: The particles distribution.
        sigma [float]: Displacement parameter.
        T [float]: Temperature.

    Returns:
        int: The number of metropolis executions that return True.
    """
    n = len(particles_distribution)
    accepted = 0
    for _ in range(n):
        index = numpy.random.randint(0, n)
        accepted += metropolis(index, particles_distribution, sigma, T)

    return accepted


@click.command()
@click.option("-n", default=12, show_default=True, help="Number of particles.")
@click.option(
    "--mcs", default=1000, show_default=True, help="Number of Monte Carlo steps."
)
@click.option(
    "--sigma", default=0.01, show_default=True, help="Displacement parameter."
)
@click.option(
    "--temp", default=0.01, show_default=True, help="Temperature (adimensional units)"
)
def main(n, mcs, sigma, temp):
    particles_distribution = random_configuration(n)

    data = pandas.DataFrame()
    for _ in tqdm(range(mcs)):
        accepted = monte_carlo_step(particles_distribution, sigma, temp)

        data = data.append(
            {
                "accepted_movements": accepted,
                "total_potential_energy": total_potential_energy(
                    particles_distribution
                ),
            },
            ignore_index=True,
        )

    print("Mean position:", numpy.mean(particles_distribution, axis=0))
    print("Total potential energy:", total_potential_energy(particles_distribution))

    if click.confirm("Do you want to save evolution?", default=True):
        fig = pyplot.figure()
        data.total_potential_energy.plot()
        pyplot.xlabel("MCS")
        pyplot.ylabel("Accepted movements")
        fig.savefig("total_potential_energy_evolution.pdf")
        pyplot.close()

        fig = pyplot.figure()
        data.accepted_movements.plot()
        pyplot.xlabel("MCS")
        pyplot.ylabel("Total potential energy")
        fig.savefig("accepted_movements_evolution.pdf")
        pyplot.close()

    if click.confirm("Do you want to save the final state?", default=True):
        plot_particle_distribution(particles_distribution).write_image(
            "final_particle_distribution.pdf"
        )


if __name__ == "__main__":
    main()
