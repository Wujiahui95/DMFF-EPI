import os
import torch
import numpy as np
import pandas as pd  # 导入 pandas 库，用于保存数据到 CSV
from sklearn.metrics import roc_auc_score, average_precision_score
from model_trans import GenomicModel

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 用于保存每个样本的结果
all_results = []
# 定义细胞系模型名称
models = ['GM12878', 'HUVEC', 'HeLa', 'IMR90', 'K562', 'NHEK']
current_model = models[4]  # 使用第一个模型

# 加载模型
embedding_matrix = np.load('embedding_matrix.npy')  # 嵌入矩阵路径
model = GenomicModel(
    embedding_matrix=embedding_matrix,
    max_len_en=3000,
    max_len_pr=2000,
    nb_words=4097,
    embedding_dim=100,
    num_heads=5
)

# 加载模型权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(f"./model/specificModel/{current_model}Model29.pth"))
model.eval()  # 设置模型为评估模式

# 测试不同细胞系
names = ['GM12878', 'HUVEC', 'HeLa', 'IMR90', 'K562', 'NHEK']
for name in names:
    # 加载测试数据
    Data_dir = f'../EPIVAN-master/Data/rawData/{name}/'
    Data_dirnpz = f'../EPIVAN-master/'
    test = np.load(Data_dir + f'{name}_test.npz')
    
    vector_en_tes = np.load(f'X_en_tes_{name}.npz')['frequencies']
    vector_pr_tes = np.load(f'X_pr_tes_{name}.npz')['frequencies']


    # 转换数据为 PyTorch tensor
    X_en_tes = torch.tensor(test['X_en_tes'], dtype=torch.long).to(device)
    X_pr_tes = torch.tensor(test['X_pr_tes'], dtype=torch.long).to(device)
    y_tes = torch.tensor(test['y_tes'], dtype=torch.float32).view(-1, 1).to(device)
    
    vector_en_tes=torch.tensor(vector_en_tes, dtype=torch.float32).to(device)
    vector_pr_tes=torch.tensor(vector_pr_tes, dtype=torch.float32).to(device)

    # 随机截取部分数据（比如 1000 条）
    num_test_samples = 1000
    test_indices = np.random.choice(len(X_en_tes), num_test_samples, replace=False)
    X_en_tes = X_en_tes[test_indices].to(device)
    X_pr_tes = X_pr_tes[test_indices].to(device)
    vector_en_tes = vector_en_tes[test_indices].to(device)
    vector_pr_tes = vector_pr_tes[test_indices].to(device)
    y_tes = y_tes[test_indices].to(device)
    


    print(f"**************** Testing {current_model} cell line specific model on {name} cell line ****************")
    # 预测
    with torch.no_grad():
        y_pred = model(X_en_tes, X_pr_tes,vector_en_tes,vector_pr_tes)['out'].cpu().numpy()
    y_tes = y_tes.cpu().numpy()

    # 计算 AUC 和 AUPR
    auc = roc_auc_score(y_tes, y_pred)
    aupr = average_precision_score(y_tes, y_pred)

    print(f"AUC : {auc:.4f}")
    print(f"AUPR : {aupr:.4f}")
# 保存每个样本的预测结果
    for i in range(len(y_tes)):
        result = {
            'Cell Line': name,
            'Sample Index': test_indices[i],
            'True Label': y_tes[i][0],  # 真实标签
            'Predicted Label': y_pred[i][0],  # 预测概率
            'Predicted Class': 1 if y_pred[i][0] > 0.5 else 0  # 预测类别（假设0.5为阈值）
        }
        all_results.append(result)

# 将所有结果保存到 CSV 文件
results_df = pd.DataFrame(all_results)
results_df.to_csv(f"test_sample_{current_model}_results.csv", index=False)

print(f"Test results saved to 'test_sample_{current_model}_results.csv'")