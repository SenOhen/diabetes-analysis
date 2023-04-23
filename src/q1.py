import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = load_diabetes(as_frame=True)

r_squared_values = {}

for item in df.feature_names:
    x = df.data[item].values.reshape(-1,1)
    y = df.target.values.reshape(-1,1)
    model = LinearRegression()
    results = model.fit(x,y)
    r_squared_values[item] = results.score(x,y)

r_squared_values = dict(sorted(r_squared_values.items(), key=lambda x:x[1],reverse=True))

plt.bar(range(len(r_squared_values)), 
list(r_squared_values.values()), tick_label=list(r_squared_values.keys()))
plt.title("Features by $R^{2}$")
plt.xlabel("Features")
plt.ylabel("$R^{2}$")
plt.savefig("figs/features_by_r_squared")
plt.close()
#plt.show()

# plt.scatter(df.data['bmi'], df.target)
# plt.show()