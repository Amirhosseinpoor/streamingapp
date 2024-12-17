import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from op.operators import Operator, NeedleVelocityOperator, NeedleAccelrationOperation,Trajectory,RelativeTrajectory
from op.operator_handler import OperatorHandler

from plot.plotting import Plotter, TwoDPlotter
import os
import time

import pandas as pd
class PlotSaver:
    path: str
    name: str
    fig: Figure

    def __init__(self, path: str, name: str, fig: Figure) -> None:
        self.path = path
        self.name = name
        self.fig = fig
        pass

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        p = self.path + "\\" + self.name + ".jpg"
        self.fig.savefig(p, dpi=300)
        # time.sleep(2)
        # self.fig.show()


class OperatorPlotHandler:
    operatorhandler: OperatorHandler
    timeSerie: np.ndarray
    plotters: list[Plotter]
    savers: list[PlotSaver]
    savePath: str

    def __init__(self, operatorhandler: OperatorHandler, timeSerie, path) -> None:
        self.operatorhandler = operatorhandler
        self.plotters = []
        self.savers = []
        self.timeSerie = timeSerie
        self.savePath = path
        for op in self.operatorhandler.operators:
            if type(op) is NeedleVelocityOperator:
                result = op.getResult()
                fig, ax = plt.subplots()
                self.plotters.append(
                    TwoDPlotter(
                        ax,
                        result[:, 0],
                        "Velocity X",
                        timeSerie,
                        "Time",
                        title="Velocity",
                    )
                )
                self.savers.append(PlotSaver(self.savePath, "Velocity X", fig))
                fig, ax = plt.subplots()
                self.plotters.append(
                    TwoDPlotter(
                        ax,
                        result[:, 1],
                        "Velocity Y",
                        timeSerie,
                        "Time",
                        title="Velocity",
                    )
                )
                self.savers.append(PlotSaver(self.savePath, "Velocity Y", fig))
                pass
            elif type(op) is NeedleAccelrationOperation:
                result = op.getResult()
                fig, ax = plt.subplots(figsize=(10, 10))

                self.plotters.append(
                    TwoDPlotter(
                        ax,
                        result[:, 0],
                        "Accelration X",
                        timeSerie,
                        "Time",
                        title="Accelration",
                    )
                )
                self.savers.append(PlotSaver(self.savePath, "Accelration X", fig))
                fig, ax = plt.subplots(figsize=(10, 10))

                self.plotters.append(
                    TwoDPlotter(
                        ax,
                        result[:, 1],
                        "Accelration Y",
                        timeSerie,
                        "Time",
                        title="Accelration",
                    )
                )
                self.savers.append(PlotSaver(self.savePath, "Accelration Y", fig))
                pass
            elif type(op) is Trajectory:
                result = op.getResult()
                fig, ax = plt.subplots(figsize=(10, 10))
                X = []
                Y = []
                for x,y in result:
                    X.append(x)
                    Y.append(y)
                self.plotters.append(
                    TwoDPlotter(
                        ax,
                        X,
                        "Trajectory X",
                        Y,
                        "Trajectory Y",
                        title="Trajectory",
                    )
                )
                self.savers.append(PlotSaver(self.savePath, "Trajectory", fig))
            elif type(op) is RelativeTrajectory:
                result = op.getResult()
                fig, ax = plt.subplots(figsize=(10, 10))
                X = []
                Y = []
                for x,y in result:
                    X.append(x)
                    Y.append(y)
                self.plotters.append(
                    TwoDPlotter(
                        ax,
                        X,
                        "Relative Trajectory X",
                        Y,
                        "Relative Trajectory Y",
                        title="Trajectory",
                    )
                )
                self.savers.append(PlotSaver(self.savePath, "Relative Trajectory", fig))
                
        pass

    def save(self):
        for saver, plotter in zip(self.savers, self.plotters):
            plotter.plot()
            # plotter.data
            
            if(type(plotter) == TwoDPlotter):
                xName = plotter.data[1]
                yName = plotter.data[3]
                x = np.array(plotter.data[0])
                y = np.array(plotter.data[2])
                x = x.astype(object)
                y = y.astype(object)

                x = np.insert(x,0,xName)
                y = np.insert(y,0,yName)
                c = np.transpose(np.concatenate([np.array([x]),np.array([y])],axis=0))
                dataFrame = pd.DataFrame(c)
                dataFrame.to_csv(saver.path + "\\" + saver.name + ".csv")
                pass
            saver.save()
        pass

    pass
