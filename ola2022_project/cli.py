import click
import logging
import numpy as np
import matplotlib.pyplot as plt

from ola2022_project import LoggingConfiguration
from ola2022_project.environment.environment import example_environment
from ola2022_project.simulation import simulation as run_simulation
from ola2022_project.learners.stupid_learner import StupidLearner

logger = logging.getLogger(__name__)


@click.group()
def main():
    """The command line interface of ola2022_project"""


@main.command()
@click.option("--n_experiments", default=1, help="number of experiments to run")
@click.option("--n_days", default=100, help="number of days for each experiment to run")
def simulation(n_experiments, n_days):
    # Setup logging, this could be changed to log to file or similar
    LoggingConfiguration()

    logger.info(
        f"Running {n_experiments} simulation{'s' if n_experiments > 1 else ''}..."
    )

    rng = np.random.default_rng()
    env = example_environment(rng=rng)
    rewards_per_experiment = run_simulation(
        rng,
        env,
        learner_factory=StupidLearner,
        prices=[10, 10, 10, 10, 10],
        n_experiment=n_experiments,
        n_day=n_days,
    )

    print(rewards_per_experiment)

    logger.info("Completed running experiments, plotting rewards...")

    _ = plt.figure()
    for i, rewards in enumerate(rewards_per_experiment):
        plt.plot(rewards, label=f"experiment {i}")

    plt.legend()
    plt.show()
