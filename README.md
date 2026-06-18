# 算法作业

本项目包含三项任务：因果推断、机器学习和深度学习。

## 项目结构

    algorithm_homework/
        -1yinguotuiduan/             # 任务1 因果推断
            -ATE_causal_analysis.py      # ATE估计值计算
            -causal_synthetic.csv
            -generate_data.py            # 生成因果数据
        -2jiqixuexi/                 # 任务2 机器学习
            -titanic.csv                 # 原始titanic数据集
            -titanic_clean.csv           # 清洗编码后titanic数据集
            -titanic_data_clean.py       # 数据清洗与特征工程：处理缺失值、生成新特征并编码
            -titanic_model.py            # 模型训练：逻辑回归、随机森林、GBDT；5折交叉验证
            -result/                     # 存放评估结果：ROC曲线、PR曲线、混淆矩阵
        -3shenduxuexi/               # 任务3 深度学习
            -cifar_train.py              # 模型训练
            -cifar_eval.py               # 模型评估
            -cifar_model.py              # 视觉模型搭建：简单CNN、ResNet、DenseNet
            -cifar_config.py             # 超参数组合
            -cifar_data.py               # 数据加载与可视化、数据增强方法
            -data/                       # CIFAR-10数据集
            -models/                     # 保存的模型权重
            -figures/                    # 训练曲线图
            -runs/                       # TensorBoard日志
        -requirements.txt            # 运行环境  
        -README.md

## 配置环境
    pip install -r requirements.txt
    
    python 版本 3.11
    pyorch 版本 12.6  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


## 任务1 因果推断
    合成因果数据，并使用合成数据估计平均处理效应（ATE）。
  ### 运行方式：
    cd 1yinguotuiduan
    python ATE_causal_analysis.py
  ### 内容：
    使用简单均值以及线性回归调整混淆变量方法，估计干预的因果效应。


## 任务2 机器学习
    使用 Titanic 数据集进行生存预测，并实现逻辑回归、随机森林、GBDT三种机器学习模型，通过5折交叉验证进行对比实验。
  ### 运行方式：
    cd 2jiqixuexi
    titanic_data_clean.py      # 缺失值处理与生成新的特征值
    python titanic_model.py    # 模型训练
  ### 输出：
    准确率、精确率、召回率、F1分数、AUC、以及各模型5折交叉验证的混淆矩阵、ROC曲线、PR曲线


## 任务3 深度学习
    在 CIFAR-10 数据集上训练图像分类模型，对比数据增强的效果。
  ### 运行方式：
    cd 3shenduxuexi
    python cifar_data.py     # 数据可视化
    python cifar_train.py    # 训练模型：SimpleCNN、ResNet、DenseNet
    python cifar_eval.py     # 评估最优模型
    tensorboard --logdir=runs # 查看tensorboard记录数据
  ### 输出：
    models/ - 保存的模型权重
    figures/ - 保存曲线对比图
    hyper_params.csv - 超参数记录
    best_model_eval_result.csv - 最优模型评估结果
