import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

diabetes = load_diabetes()

print("Computing regularization path using the LARS ...")
alphas, active, coefs = lars_path(diabetes.data, diabetes.target, method="lasso", verbose=True)

#print("alphas:", alphas)
#print("active:", active)
#print(diabetes.feature_names)
#print("coefs:", coefs)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle="dashed")
plt.xlabel("|coef| / max|coef|")
plt.ylabel("Coefficients")
plt.title("LASSO Path")
plt.axis("tight")
plt.legend([diabetes.feature_names[i] for i in active], loc="lower left", ncols=2)
plt.savefig("figs/coefficients_v_coef_over_max_coef.png")