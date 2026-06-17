import torch
# 超参数搜索组合（学习率、迭代轮数）
hyper_param_search = [
    {"lr": 1e-3, "epochs": 20},
    {"lr": 1e-3, "epochs": 30},
    {"lr": 1e-4, "epochs": 20},
    {"lr": 1e-4, "epochs": 30},
]

# 设备自动选择
decive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型保存路径
save_dir = "./models/"
# TensorBoard 日志路径
log_dir = "./runs/"