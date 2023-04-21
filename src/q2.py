import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

from q1 import df

selector = SequentialFeatureSelector(estimator=LinearRegression())
selector.fit(df)
print(selector.get_support())