import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = load_diabetes(as_frame=True)

n_components_values = []
cv_train_scores = []
cv_train_scores_std = []
cv_test_scores = []
cv_test_scores_std = []

for i in range(len(df.feature_names)):
  pcr = make_pipeline(StandardScaler(), PCA(n_components=i+1), LinearRegression())
  pcr.fit(df.data, df.target)

  validation = cross_validate(estimator=pcr, X=df.data, y=df.target, return_train_score=True)
  n_components_values.append(i+1)
  cv_train_scores.append(np.mean(validation['train_score']))
  cv_train_scores_std.append(np.std(validation['train_score']))
  cv_test_scores.append(np.mean(validation['test_score']))
  cv_test_scores_std.append(np.std(validation['test_score']))
  

plt.plot(n_components_values, cv_train_scores, label="training scores", color="blue")
plt.plot(n_components_values, np.add(np.array(cv_train_scores),np.array(cv_train_scores_std)), ls="--", color="blue")
plt.plot(n_components_values, np.subtract(np.array(cv_train_scores), np.array(cv_train_scores_std)), ls="--", color="blue")
plt.fill_between(n_components_values, np.add(np.array(cv_train_scores),np.array(cv_train_scores_std)), 
                 np.subtract(np.array(cv_train_scores), np.array(cv_train_scores_std)), alpha=0.2)
plt.plot(n_components_values, cv_test_scores, label="test scores", color="orange")
plt.plot(n_components_values, np.add(np.array(cv_test_scores), np.array(cv_test_scores_std)), ls="--", color="orange")
plt.plot(n_components_values, np.subtract(np.array(cv_test_scores), np.array(cv_test_scores_std)), ls="--", color="orange")
plt.fill_between(n_components_values, np.add(np.array(cv_test_scores), np.array(cv_test_scores_std)), 
                 np.subtract(np.array(cv_test_scores), np.array(cv_test_scores_std)), alpha=0.2)
plt.xlabel("number of components")
plt.ylabel("cross-validation scores +/- std")
plt.legend()
plt.title("Cross validation scores vs number of components using PCR")
plt.savefig("figs/cv_v_n_components_using_PCR")
plt.close()