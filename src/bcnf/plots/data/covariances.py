import matplotlib.pyplot as plt
import pandas as pd

from bcnf.plots.core import BasePlot


class DataConvariancePlot(BasePlot):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def create_plots(self, bins: int = 50) -> None:
        self.__create_covariance_plot()
        self.__create_all_pairs_plot(bins)

    def __create_covariance_plot(self) -> None:
        fig, ax = plt.subplots()

        ax.matshow(self.data.corr())
        ax.set_xticks(range(self.data.shape[1]))
        ax.set_xticklabels(self.data.columns, rotation=90)
        ax.set_yticks(range(self.data.shape[1]))
        ax.set_yticklabels(self.data.columns)

        plt.suptitle("Correlation of parameters for generated data", fontsize=16)
        plt.subplots_adjust(top=0.80, left=0.03, right=0.97, bottom=0.03)

        self.figs.append(fig)
        plt.close(fig)

    def __create_all_pairs_plot(self, bins: int = 50) -> None:
        rows = int(self.columns_count // 5)
        cols = int(self.columns_count / rows) + (self.columns_count % rows > 0)

        for i, column_i in enumerate(self.column_names):
            fig, axes = plt.subplots(nrows=rows,
                                     ncols=cols,
                                     figsize=(10, 2 * rows))
            for j, column_j in enumerate(self.column_names):
                ax = axes[j // cols, j % cols]
                ax.hist2d(self.data.iloc[:, i],
                          self.data.iloc[:, j],
                          bins=bins)
                ax.set_xlabel(column_i)
                ax.set_ylabel(column_j)

            plt.suptitle("Covariance of parameter pairs for generated data", fontsize=16)
            plt.subplots_adjust(wspace=0.7, hspace=0.5)
            plt.subplots_adjust(top=0.90, left=0.1, right=0.97, bottom=0.07)

            self.figs.append(fig)
            plt.close(fig)
