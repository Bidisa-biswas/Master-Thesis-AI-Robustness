        return joblib.load(path)
    def load_model(path):
    @staticmethod

         print(f"Model saved to {path}")
         joblib.dump(self.results, path)
             raise ValueError("Model not fitted yet.")
         if self.results is None:
    def save_model(self, path):

        return fig
            print(f"Plot saved to {save_path}")
            plt.savefig(save_path)
        if save_path:
        fig = self.results.plot()

            raise ValueError("Model not fitted yet.")
        if self.results is None:
        """Plot standardized residuals and conditional volatility."""
    def plot_fit(self, save_path=None):

        return self.results.params
            raise ValueError("Model not fitted yet.")
        if self.results is None:
        """Return model parameters."""
    def get_params(self):

        return self.results.conditional_volatility
            raise ValueError("Model not fitted yet.")
        if self.results is None:
        """Return conditional volatility."""
    def get_volatility(self):

        return self.results
        print(self.results.summary())
        self.results = self.model.fit(disp='off')
        self.model = arch_model(self.scaled_data, vol='Garch', p=self.p, o=self.o, q=self.q, dist=self.dist)

        self.scaled_data = self.data * 100
        # Rescale data by 100 for better convergence
        """Fit the GJR-GARCH model."""
    def fit(self):

        self.results = None
        self.model = None
        self.dist = dist
        self.q = q
        self.o = o
        self.p = p
        self.data = data
        """
        :param dist: Distribution to use (e.g., 'normal', 't', 'skewt')
        :param q: Lag order of lagged volatility
        :param o: Lag order of the asymmetric innovation (for GJR-GARCH)
        :param p: Lag order of the symmetric innovation
        :param data: pandas Series or DataFrame containing returns
        Initialize GJR-GARCH model.
        """
    def __init__(self, data, p=1, o=1, q=1, dist='skewt'):
class GarchModel:

import joblib
import matplotlib.pyplot as plt
import os
from arch import arch_model
import pandas as pd

