import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

from q1 import df, r_squared_values

sequential_features = []

for i in range(len(df.feature_names)-1):
    selector = SequentialFeatureSelector(estimator=LinearRegression(), n_features_to_select=i+1)
    selector.fit(df.data, df.target)
    for item in selector.get_feature_names_out():
        if item not in sequential_features:
            sequential_features.append(item)

for item in df.feature_names:
    if item not in sequential_features:
        sequential_features.append(item)

plt.bar(x = sequential_features, height=[r_squared_values[feature] for feature in sequential_features])
plt.title("Features by $R^{2}$")
plt.xlabel("Features")
plt.ylabel("$R^{2}$")
plt.savefig("figs/sfs_features_by_r_squared")
plt.close()