import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# from video import Operator,NeedleVelocityOperator,NeedleAccelrationOperation
TIME_SERIES_STYLE = "Solarize_Light2"


class Plotter:
    data: tuple[np.ndarray | str]
    title: str

    def __init__(
        self, surface: Axes, *data: tuple[np.ndarray | str], title: str
    ) -> None:
        self.data = data
        self.surface = surface
        self.title = title
        pass

    def plot(self):
        return


class TwoDPlotter(Plotter):
    def plot(self):
        print(f"{self.data[3]}  {self.data[1]}")
        self.surface.set(xlabel=self.data[3], ylabel=self.data[1])
        self.surface.plot(
            self.data[2],
            self.data[0],
        )

        # self.surface.set_title(self.title)
        # self.surface.set_xlabel(self.data[3])
        # self.surface.set_ylabel(self.data[1])
        return super().plot()
