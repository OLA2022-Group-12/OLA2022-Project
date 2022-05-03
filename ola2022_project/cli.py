import click
import logging

from ola2022_project import LoggingConfiguration

logger = logging.getLogger(__name__)


@click.group()
def main():
    """The command line interface of ola2022_project"""


# Commands can be defined as such with options and arguments
# This is where we can add commands that will call into our library code to run
# the separated learning/plotting/comparision code.


@main.command()
@click.option("--count", default=1, help="number of greetings")
@click.argument("name")
def hello(count, name):
    # Setup logging, this could be changed to log to file or similar
    LoggingConfiguration()

    logger.info(f"Running 'hello' command with arguments: count={count};name={name}")
    for x in range(count):
        click.echo(f"Hello {name}!")
