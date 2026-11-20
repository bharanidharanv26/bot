"""A single, uncluttered chart of observed reward over time."""

from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
import pandas as pd

from evaluation.metrics import reward_progress


def average_reward_chart(interactions: pd.DataFrame) -> Figure:
    progress = reward_progress(interactions)
    figure = Figure(figsize=(7, 2.8), layout="constrained", facecolor="#212121")
    axis = figure.subplots()
    axis.set_facecolor("#212121")
    if progress.empty:
        axis.text(0.5, 0.5, "No feedback yet", ha="center", va="center", transform=axis.transAxes, color="#ececec")
    else:
        axis.plot(progress["feedback_at"], progress["cumulative_average"], label="Cumulative average", marker=".", color="#ededed")
        axis.plot(progress["feedback_at"], progress["rolling_average"], label="Last 10 ratings", linestyle="--", marker=".", color="#929292")
        locator = AutoDateLocator(minticks=2, maxticks=5)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        axis.legend(frameon=False, fontsize=9, labelcolor="#c5c5c5")
    axis.set(xlabel="Feedback received (UTC)", ylabel="Average reward", ylim=(-1.1, 2.1))
    axis.set_yticks([-1, 0, 1, 2])
    axis.tick_params(colors="#aaa")
    axis.xaxis.label.set_color("#aaa")
    axis.yaxis.label.set_color("#aaa")
    axis.spines[["left", "bottom"]].set_color("#555")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.3, color="#666")
    return figure
