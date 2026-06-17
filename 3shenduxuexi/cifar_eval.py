import json
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cifar_data import get_dataloaders
from cifar_model import model_dict
from cifar_config import decive

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig_size = (6, 6)

fig_save_dir = "./figures"
os.makedirs(fig_save_dir, exist_ok=True)

best_df = pd.read_csv("hyper_params.csv", encoding="utf-8-sig")
best_df = best_df.loc[best_df.groupby(["模型名称", "增强方式"])["最优验证准确率(%)"].idxmax()].reset_index(drop=True) # 取最优

eval_result = []
with open("curve_records.json", "r", encoding="utf-8") as f:
    all_curve_data = json.load(f)
curve_df = pd.DataFrame(all_curve_data)

def evaluate_best_model(row):
    """
    加载每个模型的最优超参数模型
    row: 从最优配置表取出的一行数据
    """
    model_name = row["模型名称"]
    aug_type = row["增强方式"]
    weight_path = row["最优模型路径"]

    if not os.path.exists(weight_path):
        print(f"错误：最优模型文件不存在 -> {weight_path}")
        return None

    # 加载测试集
    _, _, test_loader = get_dataloaders(aug_type="original")
    model = model_dict[model_name]().to(decive)
    model.load_state_dict(torch.load(weight_path, map_location=decive))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(decive), labels.to(decive)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    test_acc = 100.0 * correct / total
    print(f"{model_name}模型在{aug_type}下测试集准确率: {test_acc:.2f}%")
    return test_acc

def evaluate_all_best_model():
    """评估所有最优模型"""
    for idx, row in best_df.iterrows():
        acc = evaluate_best_model(row)
        if acc is not None:
            eval_result.append({
                "模型名称": row["模型名称"],
                "增强方式": row["增强方式"],
                "最优学习率": row["最优学习率"],
                "最优训练轮数": row["最优训练轮数"],
                "最优验证准确率(%)": row["最优验证准确率(%)"],
                "测试准确率(%)": round(acc, 2),
                "最优模型路径": row["最优模型路径"]
            })

    res_df = pd.DataFrame(eval_result)
    print(res_df)
    res_df.to_csv("best_model_eval_result.csv", index=False, encoding="utf-8-sig")
    print("\n评估结果已保存至 best_model_eval_result.csv")
    return res_df


def plot_loss_acc_curve():
    """
    绘制训练/验证损失曲线和精度曲线
    :return:
    """
    model_list = best_df["模型名称"].unique()
    for model in model_list:
        current_model_best = best_df[best_df["模型名称"] == model]

        for _, best_row in current_model_best.iterrows():
            aug_type = best_row["增强方式"]
            best_lr = best_row["最优学习率"]
            best_ep = best_row["最优训练轮数"]

            match_condition = (
                (curve_df["模型名称"] == model) &
                (curve_df["增强方式"] == aug_type) &
                (curve_df["学习率"] == best_lr) &
                (curve_df["训练轮数"] == best_ep)
            )
            target_curve = curve_df[match_condition].iloc[0]

            # 取出数据
            epochs = np.arange(1, best_ep + 1)
            train_loss = target_curve["train_loss"]
            val_loss = target_curve["val_loss"]
            train_acc = target_curve["train_acc"]
            val_acc = target_curve["val_acc"]

            # 损失曲线
            plt.figure(figsize=fig_size)
            plt.plot(epochs, train_loss, label=f"{aug_type} - 训练损失", linewidth=2)
            plt.plot(epochs, val_loss, label=f"{aug_type} - 验证损失", linewidth=2, linestyle="--")

            plt.title(f"{model} 训练/验证损失曲线", fontsize=12)
            plt.xlabel("迭代轮数 Epoch", fontsize=10)
            plt.ylabel("损失值 Loss", fontsize=10)
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(fig_save_dir, f"{model}_{aug_type}_损失曲线.png")
            plt.savefig(save_path)
            plt.close()

            # 精度曲线
            plt.figure(figsize=fig_size)
            plt.plot(epochs, train_acc, label=f"{aug_type} - 训练精度", linewidth=2)
            plt.plot(epochs, val_acc, label=f"{aug_type} - 验证精度", linewidth=2, linestyle="--")

            plt.title(f"{model} 训练/验证精度曲线", fontsize=12)
            plt.xlabel("训练轮数 Epoch", fontsize=10)
            plt.ylabel("准确率 Accuracy (%)", fontsize=10)
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(fig_save_dir, f"{model}_{aug_type}_精度曲线.png")
            plt.savefig(save_path)
            plt.close()


def plot_aug_compare_val_acc():
    """
    同一模型，绘制【有无增强】的验证精度曲线
    """
    model_list = best_df["模型名称"].unique()
    for model in model_list:
        plt.figure(figsize=fig_size)
        sub_best = best_df[best_df["模型名称"] == model]
        for _, row in sub_best.iterrows():
            aug = row["增强方式"]
            lr = row["最优学习率"]
            ep = row["最优训练轮数"]
            # 匹配曲线数据
            mask = (curve_df["模型名称"] == model) & (curve_df["增强方式"] == aug) & (curve_df["学习率"] == lr) & (curve_df["训练轮数"] == ep)
            data = curve_df[mask].iloc[0]
            epochs = np.arange(1, ep+1)
            val_acc = data["val_acc"]
            plt.plot(epochs, val_acc, label=f"{aug} 验证精度", linewidth=2)
        plt.title(f"{model} 有无数据增强的验证精度对比曲线")
        plt.xlabel("Epoch")
        plt.ylabel("验证集准确率 (%)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(fig_save_dir, f"{model}_有无增强精度对比曲线.png")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    result_df = evaluate_all_best_model()
    plot_aug_compare_val_acc()
    plot_loss_acc_curve()
