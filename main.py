import click
import numpy
import pandas
from matplotlib import pyplot
from tqdm import tqdm
from utils import (
    plot_particle_distribution,
    random_configuration,
    monte_carlo_step,
    total_potential_energy,
)


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
        pyplot.ylabel("Total potential energy")
        fig.savefig("total_potential_energy_evolution.pdf")
        pyplot.close()

        fig = pyplot.figure()
        data.accepted_movements.plot()
        pyplot.xlabel("MCS")
        pyplot.ylabel("Accepted movements")
        fig.savefig("accepted_movements_evolution.pdf")
        pyplot.close()

    if click.confirm("Do you want to save the final state?", default=True):
        plot_particle_distribution(particles_distribution).write_image(
            "final_particle_distribution.pdf"
        )


if __name__ == "__main__":
    main()
