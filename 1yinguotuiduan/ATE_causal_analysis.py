import pandas as pd
import numpy as np
import statsmodels.api as sm

# 加载数据
df = pd.read_csv("causal_synthetic.csv")
T = df["T"]
Y = df["Y"]
X1 = df["X1"]
X2 = df["X2"]
print(f"真实ATE: 2.0")

# 简单均值差估计ATE
y_t1 = Y[T == 1].mean()
y_t0 = Y[T == 0].mean()
ate_simple = y_t1 - y_t0
print(f"简单均值差估计ATE: {ate_simple:.4f}")

# 线性回归调整混淆变量估计ATE
X = pd.DataFrame({"T": T, "X1": X1, "X2": X2})
X = sm.add_constant(X)  # 增加截距项
model = sm.OLS(Y, X).fit()
ate_reg = model.params["T"]
print(f"线性回归调整混淆变量后ATE: {ate_reg:.4f}")

# 输出回归摘要
print(model.summary())