import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (KFold, cross_validate, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, make_scorer)

if not os.path.exists("result"):
    os.makedirs("result")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


df = pd.read_csv("titanic.csv") # 数据加载与缺失项填充
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)

feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols.remove("Survived")

X = df[feature_cols] # 特征矩阵
y = df["Survived"] # 标签y

kf = KFold(n_splits=5, shuffle=True, random_state=42) # 5折交叉验证
# 定义评估指标字典
scoring = {
    "accuracy": make_scorer(accuracy_score),    # 准确率
    "precision": make_scorer(precision_score),  # 精确率
    "recall": make_scorer(recall_score),        # 召回率
    "f1": make_scorer(f1_score),                # F1分数
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True)  # AUC
}

# 初始化模型
lr = LogisticRegression(max_iter=5000, random_state=42, l1_ratio=0.0) # 逻辑回归
rf = RandomForestClassifier(random_state=42) # 随机森林
gbdt = XGBClassifier(random_state=42, eval_metric="logloss") # GBDT

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========任务2.2.1 超参数搜索最佳参数===========
# 逻辑回归超参数搜索
lr_param = {"C": [0.01, 0.1, 1, 10]} # C：正则化系数
lr_grid = GridSearchCV(estimator=lr, param_grid=lr_param, cv=5, scoring="roc_auc")
lr_grid.fit(X_scaled, y)
best_lr = lr_grid.best_estimator_
print(f"逻辑回归最优参数: {lr_grid.best_params_}\n 最优交叉验证AUC: {lr_grid.best_score_:.4f}")

# 随机森林超参数搜索
rf_param = {"n_estimators": [50, 100], "max_depth": [5, 10]} # n_estimators：决策树数量；max_depth：单棵树最大深度
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param, cv=5, scoring="roc_auc")
rf_grid.fit(X, y)
best_rf = rf_grid.best_estimator_
print(f"随机森林最优参数: {rf_grid.best_params_}\n 最优交叉验证AUC: {rf_grid.best_score_:.4f}")

# XGBoost超参数搜索
xgb_param = {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]} # learning_rate：学习率（步长）
xgb_grid = GridSearchCV(estimator=gbdt, param_grid=xgb_param, cv=5, scoring="roc_auc")
xgb_grid.fit(X, y)
best_xgb = xgb_grid.best_estimator_
print(f"GBDT最优参数: {xgb_grid.best_params_}\n 最优交叉验证AUC: {xgb_grid.best_score_:.4f}")


# ==========任务2.2.2 5折交叉验证===========
def evaluate_model(model, name):
    """
    5折交叉验证模型评估
    :param model: 需要评估的模型
    :param name: 需要评估的模型名称
    :return: acc准确率, pre精确率, rec召回率, f1, auc
    """
    # 用于保存每一折指标
    fold_acc, fold_pre, fold_rec, fold_f1, fold_auc = [], [], [], [], []

    # 5折交叉验证
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        fold_num = fold_idx + 1

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # 逻辑回归使用标准化数据，树模型不用
        if isinstance(model, LogisticRegression):
            X_train_trans = scaler.transform(X_train)
            X_test_trans = scaler.transform(X_test)
            model.fit(X_train_trans, y_train)
            y_pred = model.predict(X_test_trans)
            y_pred_prob = model.predict_proba(X_test_trans)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 计算当前折指标
        acc_fold = accuracy_score(y_test, y_pred)
        pre_fold = precision_score(y_test, y_pred)
        rec_fold = recall_score(y_test, y_pred)
        f1_fold = f1_score(y_test, y_pred)
        auc_fold = roc_auc_score(y_test, y_pred_prob)

        fold_acc.append(acc_fold)
        fold_pre.append(pre_fold)
        fold_rec.append(rec_fold)
        fold_f1.append(f1_fold)
        fold_auc.append(auc_fold)

        print(f"\n第{fold_num}折指标：")
        print(f"准确率:{acc_fold:.4f}  精确率:{pre_fold:.4f}  召回率:{rec_fold:.4f}  F1:{f1_fold:.4f}  AUC:{auc_fold:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name}的第{fold_num}折混淆矩阵")
        plt.tight_layout()
        plt.savefig(f"result/{name}_第{fold_num}折_混淆矩阵.png")
        plt.close()

        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label="ROC曲线")
        plt.xlabel("FPR(假正例率)")
        plt.ylabel("TPR(真正例率)")
        plt.title(f"{name}的第{fold_num}折ROC曲线")
        plt.legend()
        plt.savefig(f"result/{name}_第{fold_num}折_ROC曲线.png")
        plt.close()

        # 绘制PR曲线
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.plot(recall, precision, label="PR曲线")
        plt.xlabel("Recall(召回率)")
        plt.ylabel("Precision(精确率)")
        plt.title(f"{name}的第{fold_num}折PR曲线")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"result/{name}_第{fold_num}折_PR曲线.png")
        plt.close()

    acc = np.mean(fold_acc)
    pre = np.mean(fold_pre)
    rec = np.mean(fold_rec)
    f1 = np.mean(fold_f1)
    auc = np.mean(fold_auc)

    print(f"\n {name}的平均指标")
    print(f"准确率: {acc:.4f}  精确率: {pre:.4f}  召回率: {rec:.4f}  F1值: {f1:.4f}  ROC-AUC: {auc:.4f}")

    return acc, pre, rec, f1, auc

# 评估模型
evaluate_model(best_lr, "逻辑回归")
evaluate_model(best_rf, "随机森林")
evaluate_model(best_xgb, "GBDT")


# ==========任务2.2.3 输出特征重要性（对树模型）和特征系数（对逻辑回归）=============
lr_coef = pd.Series(best_lr.coef_[0], index=feature_cols).sort_values(ascending=False)
print("逻辑回归特征系数:", lr_coef)

rf_imp = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("随机森林特征重要性: ", rf_imp)

xgb_imp = pd.Series(best_xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("GBDT特征重要性: ", xgb_imp)


# ==========任务2.2.4 投票集成和Stacking集成=============
vote_clf = VotingClassifier(
    estimators=[("lr", best_lr), ("rf", best_rf), ("xgb", best_xgb)],
    voting="soft"
) # soft：软投票
evaluate_model(vote_clf, "软投票集成")

# Stacking堆叠
stack_clf = StackingClassifier(
    estimators=[("lr", best_lr), ("rf", best_rf), ("xgb", best_xgb)], # 第一层：三个基础模型
    final_estimator=LogisticRegression() # 第二层元模型
)
evaluate_model(stack_clf, "Stacking堆叠集成")