import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, f
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

class lm_manual:
    def __init__(self, x, y):
        
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x and y must be 1D arrays of equal length")
        mask = np.isfinite(x) & np.isfinite(y)
        self.independent = x[mask]
        self.dependent = y[mask]
        self.n = len(self.dependent)
        if self.n < 3:
            raise ValueError("At least 3 valid observations required")
        if len(np.unique(self.independent)) < 2:
            raise ValueError("X-values must have variation")
        if len(np.unique(self.dependent)) < 2:
            raise ValueError("Y-values must have variation")
        
        # Mean calculations
        sum_x = 0.0
        sum_y = 0.0
        for num_x, num_y in zip(self.independent, self.dependent):
            sum_x += num_x
            sum_y += num_y
        self.mean_x = sum_x / len(self.independent)
        self.mean_y = sum_y / len(self.dependent)

        # Slope calculation
        self.sxx = 0.0
        self.sxy = 0.0
        for num_x, num_y in zip(self.independent, self.dependent):
            self.sxy += (num_x - self.mean_x) * (num_y - self.mean_y)
            self.sxx += (num_x - self.mean_x) ** 2
        self.beta_1 = self.sxy / self.sxx
        
        # Intercept calculation
        self.beta_0 = self.mean_y - (self.beta_1 * self.mean_x)

        # Summary calculations

        # Fitted y-values
        self.fitted_y = self.beta_0 + (self.beta_1 * self.independent)
        
        # Residuals    
        self.residuals = self.dependent - self.fitted_y
        
        # RSS
        self.rss = np.sum(self.residuals ** 2)

        # TSS
        self.tss = np.sum((self.dependent - self.mean_y) ** 2)

        # Degrees of freedom
        self.df = self.n - 2

        # R-Squared
        self.R_Squared = 1.0 - (self.rss / self.tss)

        # Adjusted R-Squared
        self.adjusted_r_squared = 1 - ((1 - self.R_Squared) * (self.n - 1) / self.df)

        # Residual standard error
        self.rse = (self.rss / self.df) ** 0.5

        # F-Statistic
        self.f_stat = (self.tss - self.rss) / (self.rss / self.df)

        # Model P-value
        self.p_value = f.sf(self.f_stat, 1, self.df)

        # Coefficient standard errors
        self.se_beta_1 = self.rse / (self.sxx ** 0.5)
        self.se_beta_0 = self.rse * ((1 / self.n) + (self.mean_x ** 2 / self.sxx)) ** 0.5

        # Coefficient T-statistics
        self.t_beta_0 = self.beta_0 / self.se_beta_0
        self.t_beta_1 = self.beta_1 / self.se_beta_1

        # Coefficient p-values
        self.p_beta_0 = 2 * t.sf(abs(self.t_beta_0), self.df)
        self.p_beta_1 = 2 * t.sf(abs(self.t_beta_1), self.df)

        # Coefficient confidence intervals
        t_critical = t.ppf(1 - 0.05 / 2, self.df)
        self.ci_beta_0 = (self.beta_0 - t_critical * self.se_beta_0, self.beta_0 + t_critical * self.se_beta_0)
        self.ci_beta_1 = (self.beta_1 - t_critical * self.se_beta_1, self.beta_1 + t_critical * self.se_beta_1)

        # Diagnostics

        # Leverage
        self.leverage = (1 / self.n) + ((self.independent - self.mean_x) ** 2 / self.sxx)

        # Standardized residuals
        if self.rse > 0 and np.all(self.leverage < 1):
            self.std_residuals = self.residuals / (self.rse * np.sqrt(1 - self.leverage))

            # Scale-location
            self.scale_location = np.sqrt(np.abs(self.std_residuals))

            # Cook's distance
            self.cooks_distance = ((self.std_residuals ** 2 / 2) * (self.leverage / (1 - self.leverage)))

        else:
            self.std_residuals = np.full(self.n, np.nan)
            self.scale_location = np.full(self.n, np.nan)
            self.cooks_distance = np.full(self.n, np.nan)
        
    def summary(self):
        print("Linear Regression Summary")
        print("-------------------------")
        print(f"Observations: {self.n}")
        print(f"R-Squared: {self.R_Squared:.4f}")
        print(f"Adjusted R-Squared: {self.adjusted_r_squared:.4f}")
        print(f"Residual Standard Error: {self.rse:.4f}")
        print(f"F-Statistic: {self.f_stat:.4f}")
        print(f"Model P-Value: {self.p_value:.4g}")
        print("\nCoefficients:")
        print(f"Intercept: {self.beta_0:.4f}")
        print(f"Slope: {self.beta_1:.4f}")
        print("\nStandard Errors:")
        print(f"Intercept: {self.se_beta_0:.4f}")
        print(f"Slope: {self.se_beta_1:.4f}")
        print("\nT-Statistics:")
        print(f"Intercept: {self.t_beta_0:.4f}")
        print(f"Slope: {self.t_beta_1:.4f}")
        print("\nCoefficient P-Values:")
        print(f"Intercept: {self.p_beta_0:.4g}")
        print(f"Slope: {self.p_beta_1:.4g}")
        print("\n95% Confidence Intervals:")
        print(f"Intercept: ({self.ci_beta_0[0]:.4f}, {self.ci_beta_0[1]:.4f})")
        print(f"Slope: ({self.ci_beta_1[0]:.4f}, {self.ci_beta_1[1]:.4f})")

    def predict(self, x_value, interval=None, confidence=0.95):
        if interval is not None:
            interval = interval.upper()
        if interval is not None and interval not in ("PREDICTION", "CONFIDENCE"):
             raise ValueError("Interval must be Prediction or Confidence")
        x_value = np.atleast_1d(np.asarray(x_value, dtype=float))
        predictions = self.beta_0 + self.beta_1 * x_value
        if interval is None:
            return predictions
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        alpha = 1 - confidence
        t_critical = t.ppf(1 - alpha / 2, self.df)
        lower = np.array([])
        upper = np.array([])
        for x, predicted_y in zip(x_value, predictions):
            if interval == "CONFIDENCE":
                se = self.rse * np.sqrt((1 / self.n) + ((x - self.mean_x) ** 2 / self.sxx))
            elif interval == "PREDICTION":
                se = self.rse * np.sqrt(1 + (1 / self.n) + ((x - self.mean_x) ** 2 / self.sxx))
            margin = t_critical * se
            lower = np.append(lower, predicted_y - margin)
            upper = np.append(upper, predicted_y + margin)
        return pd.DataFrame({"X": x_value,"Prediction": predictions,"Lower Bound": lower,"Upper Bound": upper}).round(4)
        
    def plot_regression(self, x_label="Independent Variable (X)", y_label="Dependent Variable (Y)", title="Simple Linear Regression"):
        sorted_indices = np.argsort(self.independent)
        x_sorted = self.independent[sorted_indices]
        y_sorted = self.fitted_y[sorted_indices]
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.independent, y=self.dependent, label="Actual Values")
        sns.lineplot(x=x_sorted, y=y_sorted, label="Regression Line")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_diagnostics(self):
        if not np.all(np.isfinite(self.std_residuals)):
            raise ValueError("Diagnostics unavailable for this model")

        fig, ax = plt.subplots(2, 2, figsize=(12, 9))

        # 1. Residuals vs Fitted
        sns.scatterplot(x=self.fitted_y, y=self.residuals, ax=ax[0, 0])
        smooth = lowess(self.residuals, self.fitted_y)
        ax[0, 0].plot(smooth[:, 0], smooth[:, 1], color="red")
        ax[0, 0].axhline(0, color="gray", linestyle="--")
        ax[0, 0].set_title("Residuals vs Fitted")
        ax[0, 0].set_xlabel("Fitted Values")
        ax[0, 0].set_ylabel("Residuals")

        # 2. Normal Q-Q 
        stats.probplot(self.std_residuals, dist="norm", plot=ax[0, 1])
        ax[0, 1].set_title("Normal Q-Q")

        # 3. Scale-Location
        sns.scatterplot(x=self.fitted_y, y=self.scale_location, ax=ax[1, 0])
        smooth = lowess(self.scale_location, self.fitted_y)
        ax[1, 0].plot(smooth[:, 0], smooth[:, 1], color="red")
        ax[1, 0].set_title("Scale-Location")
        ax[1, 0].set_xlabel("Fitted Values")
        ax[1, 0].set_ylabel("√|Standardized Residuals|")

        # 4. Residuals vs Leverage
        sns.scatterplot(x=self.leverage, y=self.std_residuals, ax=ax[1, 1])
        ax[1, 1].axhline(0, color="gray", linestyle="--")

        # Cook's distance contours
        h = np.linspace(max(0.000001, self.leverage.min() * 0.8),min(0.999, self.leverage.max() * 1.2),200)

        for d in [0.5, 1.0]:
            bound = np.sqrt(d * 2 * (1 - h) / h)
            ax[1, 1].plot(h, bound, "r--", alpha=0.5)
            ax[1, 1].plot(h, -bound, "r--", alpha=0.5, label=f"Cook's D = {d}")

        limit = max(3, np.max(np.abs(self.std_residuals)) * 1.2)
        ax[1, 1].set_ylim(-limit, limit)
        ax[1, 1].set_title("Residuals vs Leverage")
        ax[1, 1].set_xlabel("Leverage")
        ax[1, 1].set_ylabel("Standardized Residuals")
        ax[1, 1].legend()
        plt.tight_layout()
        plt.show()

        
