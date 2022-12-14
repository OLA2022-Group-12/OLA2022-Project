import io

import numpy as np
from graphviz import Digraph
from PIL import Image
from matplotlib.colors import BASE_COLORS

from ola2022_project.utils import add_headers_to_plot
from ola2022_project.environment.environment import EnvironmentData, alpha_function


def show_class_ratios(env: EnvironmentData, ax):
    """Plot class ratios to given Axes

    Arguments:
        env: environment to plot

        ax: matplotlib Axes to plot to
    """

    n_classes = len(env.classes_parameters)

    class_names = [f"Class {i}" for i in range(n_classes)]
    class_colors = [color for color, _ in zip(BASE_COLORS.values(), range(n_classes))]

    ax.pie(env.class_ratios, labels=class_names, colors=class_colors)
    ax.set_title("Class ratios")


def show_alpha_ratios(env: EnvironmentData, fig, combine_rows=True):
    """Plot alpha ratios to given Figure

    Arguments:
        env: environment to plot

        fig: figure to plot data to

        combine_rows: wether to combine classes in same graph/row
    """

    n_products = len(env.product_prices)
    n_classes = len(env.classes_parameters)
    n_budget_steps = 10
    budget_steps = np.linspace(0, env.total_budget, n_budget_steps, endpoint=True)

    class_names = [f"Class {i}" for i in range(n_classes)]
    class_colors = [color for color, _ in zip(BASE_COLORS.values(), range(n_classes))]

    row_label = "Ratio"
    col_label = "Budget"

    n_columns = n_products
    col_headers = [f"Product {i + 1}" for i in range(n_products)]

    if combine_rows:
        n_rows = 1
        row_headers = [""]

        axs = fig.subplots(1, n_columns, sharey=True, squeeze=True)
        axss = [axs] * n_classes
    else:
        n_rows = n_classes + 1
        row_headers = class_names + ["Aggregated"]

        axss = fig.subplots(n_rows, n_columns, sharey=True, sharex=True, squeeze=False)

    aggregated = np.zeros((n_products, n_budget_steps))

    for user_class, (class_parameters, class_ratio, axs) in enumerate(
        zip(env.classes_parameters, env.class_ratios, axss)
    ):
        for product, (class_product_params, ax) in enumerate(
            zip(class_parameters, axs)
        ):
            alpha_ratios = alpha_function(
                budget_steps, class_product_params.upper_bound, env.total_budget
            )

            aggregated[product] += class_ratio * alpha_ratios

            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(row_label)

            ax.set_ylim(0, 1)
            ax.set_xlim(0, env.total_budget)
            ax.set_xticks([0, env.total_budget])

            ax.plot(
                budget_steps,
                alpha_ratios,
                color=class_colors[user_class],
                label=f"Class {user_class}",
            )

    for product, ax in enumerate(axss[-1]):
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(row_label)
        ax.set_xlabel(col_label)

        ax.plot(budget_steps, aggregated[product], color="black", label="Aggregated")

    add_headers_to_plot(
        axss,
        col_headers=col_headers,
        row_headers=row_headers,
    )
    fig.suptitle("Alpha functions or alpha ratios given budget")
    fig.tight_layout()

    if combine_rows:
        axss[-1][-1].legend()


def show_product_graph(env: EnvironmentData, ax):
    """Plot the graph of products to the given Axes from matplotlib

    Arguments:
        env: environment to plot

        ax: matplotlib Axes to plot to
    """
    n_products = len(env.product_prices)
    product_graph = Digraph("product_graph")

    for product in range(n_products):
        product_graph.node(str(product + 1))

    for product, next_products in enumerate(env.next_products):
        for next_product in next_products:
            product_graph.edge(
                str(product + 1),
                str(next_product + 1),
                label=f"{env.graph[product, next_product]:.2f}",
            )

    image = Image.open(io.BytesIO(product_graph.pipe(format="png")))

    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Product Graph")
