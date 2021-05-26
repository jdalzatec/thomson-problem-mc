import time

import numpy
import pandas
import streamlit

from utils import (
    monte_carlo_step,
    plot_particle_distribution,
    random_configuration,
    total_potential_energy,
)

streamlit.header("Thomson problem")
streamlit.write(
    """
    This app lets you solve the
    [Thomson problem](https://en.wikipedia.org/wiki/Thomson_problem) numerically by
    using the [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method).
    Also we use the [metropolis algorithm](https://www.sciencedirect.com/topics/computer-science/metropolis-algorithm)
    to relax the system. To look into the code, please visit [this repository](https://github.com/jdalzatec/thomson-problem-mc).
"""
)
with streamlit.form(key="form"):
    col1, col2 = streamlit.beta_columns(2)
    with col1:
        num_particles = streamlit.number_input(
            "Num. particles", min_value=2, max_value=32, value=5
        )

    with col2:
        mcs = streamlit.number_input("MCS", min_value=1, value=500)

    sigma = streamlit.slider(
        "sigma", min_value=0.1, max_value=1.0, step=0.001, format="%.3f"
    )
    temp = streamlit.slider(
        "Temperature", min_value=0.001, max_value=10.0, format="%.3f"
    )
    run_button = streamlit.form_submit_button("Run")


progress_bar = streamlit.progress(0)
mcs_text = streamlit.empty()
plot_section = streamlit.empty()

if run_button:
    accepted_section, energy_section = streamlit.beta_columns(2)
    with accepted_section:
        streamlit.subheader("Accepted movements")
        accepted_chart = streamlit.line_chart([])

    with energy_section:
        streamlit.subheader("Potential energy")
        energy_chart = streamlit.line_chart([])


if run_button:
    particles_distribution = random_configuration(num_particles)

    data = pandas.DataFrame()
    for i in range(mcs):
        accepted = monte_carlo_step(particles_distribution, sigma, temp)
        energy = total_potential_energy(particles_distribution)

        data = data.append(
            {"accepted_movements": accepted, "total_potential_energy": energy},
            ignore_index=True,
        )

        progress_bar.progress((i + 1) / mcs)
        mcs_text.write(f"Currently at {i + 1} step")
        plot_section.plotly_chart(plot_particle_distribution(particles_distribution))
        accepted_chart.add_rows([accepted])
        energy_chart.add_rows([energy])

    streamlit.dataframe(data)
    streamlit.subheader("Mean position")
    streamlit.write(numpy.mean(particles_distribution, axis=0))
    streamlit.subheader(f"Total potential energy for {num_particles} particles")
    streamlit.write(total_potential_energy(particles_distribution))

    streamlit.balloons()
