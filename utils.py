from plotly import graph_objects


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
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig
