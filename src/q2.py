import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector

df = load_diabetes(as_frame=True)