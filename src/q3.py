import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import statsmodels.api as sm
import matplotlib.pyplot as plt

from q1 import df, r_squared_values
from q2 import sequential_features

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].bar(range(len(r_squared_values)), list(r_squared_values.values()), tick_label=list(r_squared_values.keys()))
axs[0].set_title("Features by $R^{2}$, regular")
axs[0].set(xlabel="Features", ylabel ="$R^{2}$")

axs[1].bar(x = sequential_features, height=[r_squared_values[feature] for feature in sequential_features])
axs[1].set_title("Features by $R^{2}$, sequentially selected")
axs[1].set(xlabel="Features", ylabel ="$R^{2}$")
plt.savefig("figs/regular_v_sequentially_selected.png")
plt.close()