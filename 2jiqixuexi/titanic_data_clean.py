import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# df = sns.load_dataset("titanic") #加载数据集
df = pd.read_csv("titanic.csv")
# print("原始数据形状:", df.shape)
# print("列表表头:", df.columns.tolist())
# print("缺失值统计:", df[df.columns.tolist()].isnull().sum())

# 筛选存在缺失的列
mis_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist() #['Age', 'Cabin', 'Embarked']
print("存在缺失值的列:", mis_cols)


# =========任务2.1.1 处理缺失值========
def fill_compare(data):
    """
    处理年龄Age项缺失值，纯数值，分别采用均值填充、中位数填充以及众数填充，并使用交叉验证选最优
    座位号Cabin缺失687，占比很多，直接整列删除不填充
    登船码头Embarked仅缺失2项，采用众数填充
    :param data: titanic数据集
    :return: 最优缺失值处理方法
    """
    df_tmp = data.copy()
    # Embarked众数填充、Cabin直接删除
    df_tmp["Embarked"] = df_tmp["Embarked"].fillna(df_tmp["Embarked"].mode()[0])
    df_tmp.drop("Cabin", axis=1, inplace=True)

    X_raw = df_tmp.drop(["Survived"], axis=1).select_dtypes(include=[np.number]) # 只筛选数值型特征
    y = df_tmp["Survived"] # 分类标签-是否幸存

    # -------- 1 Age均值填充 --------
    df1 = df_tmp.copy()
    df1["Age"] = df1["Age"].fillna(df1["Age"].mean())
    X1 = df1[X_raw.columns]
    score1 = cross_val_score(LogisticRegression(max_iter=1000), X1, y, cv=5).mean() # 5折交叉验证

    # -------- 2 Age中位数填充 --------
    df2 = df_tmp.copy()
    df2["Age"] = df2["Age"].fillna(df2["Age"].median())
    X2 = df2[X_raw.columns]
    score2 = cross_val_score(LogisticRegression(max_iter=1000), X2, y, cv=5).mean()

    # -------- 3 Age众数填充 --------
    df3 = df_tmp.copy()
    df3["Age"] = df3["Age"].fillna(df3["Age"].mode()[0])
    X3 = df3[X_raw.columns]
    score3 = cross_val_score(LogisticRegression(max_iter=1000), X3, y, cv=5).mean()

    print(f"Age均值填充mean准确率: {score1:.4f}")
    print(f"Age中位数填充median准确率: {score2:.4f}")
    print(f"Age众数填充mode准确率: {score3:.4f}")

    # 对比分数，返回最优的Age填充策略
    score_dict = {"mean": score1, "median": score2, "mode": score3}
    best_strategy = max(score_dict, key=score_dict.get)
    # print(best_strategy)

    return best_strategy


best_age_fill = fill_compare(df)
print(f"\n经过5折交叉验证，Age最优填充策略为：{best_age_fill}")

# 执行最优填充策略
# 年龄Age均值填充, 座位号Cabin整列删除不填充, 登船码头Embarked众数填充
df_clean = df.copy()
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])
df_clean.drop("Cabin", axis=1, inplace=True)


# =========任务2.1.2 构造至少6个新特征========
df_clean["Family_Size"] = df_clean["SibSp"] + df_clean["Parch"] + 1  # 家庭规模 = 兄弟姐妹(SibSp) + 父母子女(Parch) + 自己
df_clean["Is_Alone"] = (df_clean["Family_Size"] == 1).astype(int) # 是否独自旅行；家庭规模 = 1 表示孤身一人
df_clean["Title"] = df_clean["Name"].str.extract(r' ([A-Za-z]+)\.')[0] # 提取称呼：Mr、Mrs、Miss等
df_clean["Is_Adult"] = (df_clean["Age"] >= 18).astype(int) # 是否成年人；年龄 >= 18为成年人，标记为 1
df_clean["Embarked_Code"] = df_clean["Embarked"].map({"S": 0, "C": 1, "Q": 2}) # 将登船码头S、C、Q映射为数值，可用于判断生存率是否受登船码头影响
df_clean["Has_SibSp"] = (df_clean["SibSp"] > 0).astype(int) # 是否有兄弟姐妹随行：1表示有， 0表示没有
df_clean["Has_Parch"] = (df_clean["Parch"] > 0).astype(int) # 是否有父母子女随行：1表示有， 0表示没有



# =========任务2.1.3 编码类别特征，使用one-hot编码 ==========
# 需要编码的分类特征：性别Sex、登船码头Embarked、新增称呼Title
cat_cols = ["Sex", "Embarked", "Title"]
df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=False, dtype=int)



df_clean.to_csv("titanic_clean.csv", index=False)
print("\n预处理完成，数据已保存为 titanic_clean.csv")

