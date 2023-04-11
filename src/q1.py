import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = load_diabetes(as_frame=True)
#print(df.data['age'])
#print(df.feature_names)

r_squared_values = {}

# for item in df.feature_names:
#     model = sm.OLS(df.target,df.data[item])
#     results = model.fit()
#     r_squared_values[item] = results.rsquared

for item in df.feature_names:
    x = df.data[item]
    y = df.target
    model = LinearRegression()
    results = model.fit(x,y)
    r_squared_values[item] = results.score()

sorted_r_squared_values = dict(sorted(r_squared_values.items(), key=lambda x:x[1],reverse=True))

print(sorted_r_squared_values)

plt.bar(range(len(sorted_r_squared_values)), 
list(sorted_r_squared_values.values()), tick_label=list(sorted_r_squared_values.keys()))
plt.title("Features by $R^{2}$")
plt.xlabel("Features")
plt.ylabel("$R^{2}$")
plt.savefig("figs/features_by_r_squared")
plt.show()

# plt.scatter(df.data['bmi'], df.target)
# plt.show()