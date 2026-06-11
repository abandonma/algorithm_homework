import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

# 混淆变量
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)

# 干预变量 T（倾向得分依赖于 X1, X2）
propensity = 1 / (1 + np.exp(-(0.5*X1 + 0.5*X2)))
T = np.random.binomial(1, propensity, n)

# 结果变量 Y，真实处理效应为常数 2.0
Y0 = 1 + X1 + X2 + np.random.normal(0, 0.5, n)
Y1 = Y0 + 2.0
Y = T * Y1 + (1 - T) * Y0

df = pd.DataFrame({'X1': X1, 'X2': X2, 'T': T, 'Y': Y})
df.to_csv('causal_synthetic.csv', index=False)