# OLA2022-Project

Project for the Online Learning Applications class at Politecnico di Milano a.a. 2021-2022

## Development instructions

See [the documentation of poetry](https://python-poetry.org/docs/basic-usage/)
for instructions on how to setup the virtual environment with all the needed
python packages. It enables proper dependecy management in python.

After the virtual environment is setup and initialized with `poetry install`,
the jupyter notebooks server can be started with `poetry run jupyter notebook`.
This will enable quick prototyping and utilization of the code created in the
library. In the notebook, code from our library can then be used by importing
with `from ola2022_project import <name of class/function>` or similar. The
notebooks can be found in the `notebooks` folder.

Also to ensure that formatting and git quality remains high,
[`pre-commit`](https://pre-commit.com), a framework for git pre-commit hooks, is
used. This means that whenever code is checked into the project some checks will
run to ensure it is formatted correctly and prevent erronous git commits. To
setup this, `poetry run pre-commit install` has to be run. _If there every comes
a time you want to commit without the pre-commit hooks running, `git commit
--no-verify <args>` can be used_.

The tests of the project can be run using `poetry run pytest`. A dummy test can
be found in `tests/test_version.py`.

Lastly, as we will be implemeting different steps and learning procedures, and
idea is to separate this into separate click commands. See
[clicks homepage](https://click.palletsprojects.com/en/8.1.x/) for more
information about the library, though the most essential information can be
found in `ola2022_project/cli.py`, which already contains a "Hello world"
command as an example. To run the cli of the project use `poetry run
ola2022_project`.
