import click
import logging
import numpy as np
import matplotlib.pyplot as plt

from ola2022_project import LoggingConfiguration
from ola2022_project.environment.environment import example_environment, Step
from ola2022_project.simulation.simulation import simulation as run_simulation
from ola2022_project.learners import (  # noqa
    ClairvoyantLearner,
    StupidLearner,
    GraphlessLearner,
)

logger = logging.getLogger(__name__)


@click.group()
def main():
    """The command line interface of ola2022_project"""


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="enable debug loggin")
@click.option("--n_experiments", default=1, help="number of experiments to run")
@click.option("--n_days", default=100, help="number of days for each experiment to run")
@click.option(
    "--show_progress_graphs",
    "-s",
    is_flag=True,
    help="show graphs of the learning progress",
)
def simulation(verbose, n_experiments, n_days, show_progress_graphs):
    # Setup logging, this could be changed to log to file or similar
    LoggingConfiguration("DEBUG" if verbose else "INFO")

    logger.info(
        f"Running {n_experiments} simulation{'s' if n_experiments > 1 else ''}..."
    )

    rng = np.random.default_rng()
    env = example_environment(rng=rng)
    rewards_per_experiment = run_simulation(
        rng=rng,
        env=env,
        learner_factory=GraphlessLearner,
        n_customers_mean=10,
        n_customers_variance=1,
        n_days=n_days,
        n_experiment=n_experiments,
        step=Step.ZERO,
        show_progress_graphs=show_progress_graphs,
    )
    logger.debug(f"Rewards per experiment: {rewards_per_experiment}")

    logger.info("Completed running experiments, plotting rewards...")
    _ = plt.figure()
    for i, rewards in enumerate(rewards_per_experiment):
        plt.plot(rewards, label=f"experiment {i}")

    plt.legend()
    plt.show()
