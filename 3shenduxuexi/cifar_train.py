import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cifar_data import get_dataloaders
from cifar_model import model_dict
from cifar_config import hyper_param_search, decive, save_dir, log_dir


os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

best_record = []
curve_records = []
curve_save_path = "curve_records.json"

def train_model(model_name, aug_type, lr, epochs):
    """
    训练模型
    :param model_name: 模型名称 SimpleCNN、ResNet、DenseNet
    :param aug_type: 增强方式 随机裁剪、 随机水平翻转、颜色抖动、 cutout遮挡
    :param lr: 学习率
    :param epochs: 训练轮次
    :return: train_loss_list, val_loss_list, val_acc_list
    """

    train_loader, val_loader, _ = get_dataloaders(aug_type=aug_type)
    model = model_dict[model_name]().to(decive)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.8) #学习率衰减

    # TensorBoard日志命名
    log_path = f"{log_dir}_{model_name}_{aug_type}_lr{lr}_epochs{epochs}"
    writer = SummaryWriter(log_dir=log_path)

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []


    total_start_time = time.time() # 计时

    print(f"\n===== 开始训练： 模型:{model_name}  增强方法:{aug_type}  学习率:{lr}  训练轮次:{epochs} =====")

    for epoch in range(epochs):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(decive), labels.to(decive)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for imgs, labels in pbar_val:
                imgs, labels = imgs.to(decive), labels.to(decive)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total

        # 更新学习率
        scheduler.step()

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        # 写入日志
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)

        epoch_cost = time.time() - epoch_start

        # print(f"Epoch:{epoch+1:2d}  train_loss:{train_loss:.4f}  val_loss:{val_loss:.4f}  train_acc:{train_acc:.2f}%  val_acc:{val_acc:.2f}%   本轮耗时: {epoch_cost:.2f} s")

    # 模型总耗时
    model_total_time = time.time() - total_start_time
    max_val_acc = max(val_acc_list)
    print(f"\n{model_name} + {aug_type}:  学习率:{lr},  训练轮次:{epochs}, val准确率：{max_val_acc:.2f}%， 耗时: {model_total_time:.2f} 秒\n")

    # 保存模型权重
    save_name = f"{save_dir}_{model_name}_{aug_type}_lr{lr}_epochs{epochs}.pth"
    torch.save(model.state_dict(), save_name)
    # print(f"模型权重已保存至: {save_name}\n")

    writer.close()
    curve_records.append({
        "模型名称": model_name,
        "增强方式": aug_type,
        "学习率": lr,
        "训练轮数": epochs,
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_acc": train_acc_list,
        "val_acc": val_acc_list
    })

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list, max_val_acc, save_name

if __name__ == "__main__":
    print(f"运行设备: {decive}")
    model_names = ["SimpleCNN", "ResNet", "DenseNet"]
    aug_list = ["original", "aug"]
    hp_list = hyper_param_search

    for model_name in model_names:
        for aug in aug_list:
            print(f"开始搜索模型:{model_name} 在 增强方式:{aug} 下的最优超参数组合")
            best_acc = -1.0
            best_hp = None
            best_path = ""

            for hp in hp_list:
                lr = hp["lr"]
                epochs = hp["epochs"]
                _, _, _,_, max_val_acc, save_name = train_model(model_name, aug, lr, epochs)

                if max_val_acc > best_acc:
                    best_acc = max_val_acc
                    best_hp = hp
                    best_path = save_name

                best_record.append({
                    "模型名称": model_name,
                    "增强方式": aug,
                    "最优学习率": best_hp["lr"],
                    "最优训练轮数": best_hp["epochs"],
                    "最优验证准确率(%)": round(best_acc, 2),
                    "最优模型路径": best_path
                })

                print(f"{model_name} + {aug}: 最优超参：lr={best_hp['lr']}, epochs={best_hp['epochs']}, 最优验证精度：{best_acc:.2f}%\n")

    with open(curve_save_path, "w", encoding="utf-8") as f:
        json.dump(curve_records, f, ensure_ascii=False, indent=2)

    df_best = pd.DataFrame(best_record)
    df_best.to_csv("hyper_params.csv", index=False, encoding="utf-8-sig")
    print("最优超参已保存至 hyper_params.csv")
    print(df_best)



