from plotly import graph_objects
import numpy


def plot_particle_distribution(particles_distribution):
    """Creates a figure with the particles distribution.

    Arguments:
        state [numpy.ndarray]: Array with the particles distribution. If you have N
            particles, then the shape of state is (N, 3).

    Returns:
        graph_objects.Figure: The figure with the state.
    """

    x, y, z = particles_distribution.T

    fig = graph_objects.Figure(
        data=[
            graph_objects.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=12,
                    color="crimson",
                    opacity=0.8,
                ),
            )
        ],
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene_aspectmode="cube",
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[-1.2, 1.2],
            ),
            yaxis=dict(
                nticks=4,
                range=[-1.2, 1.2],
            ),
            zaxis=dict(
                nticks=4,
                range=[-1.2, 1.2],
            ),
        ),
    )
    return fig


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
