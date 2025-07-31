import os
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_trans import GenomicModel

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 自定义回调类，用于在每个 epoch 结束后计算 AUC 和 AUPR
class RocCallback:
    def __init__(self, val_loader, name, model,device, save_path="./model/specificModel/"):
        """
        初始化回调类。
        :param val_data: 验证集数据，包含 (增强子, 启动子, 标签)
        :param name: 当前模型名称
        :param model: 模型对象
        :param save_path: 模型保存路径
        """
        self.val_loader = val_loader
        # self.en = val_data[0]
        # self.pr = val_data[1]
        # self.vecen = val_data[2]
        # self.vecpr = val_data[3]
        # self.y = val_data[4]
        self.name = name
        self.model = model
        self.device = device
        self.save_path = save_path

    def on_epoch_end(self, epoch):
        """
        每个 epoch 结束时计算验证集的 AUC 和 AUPR，并保存模型权重。
        :param epoch: 当前 epoch 编号
        """
        # self.model.eval()
        # with torch.no_grad():
        #     y_pred = self.model(self.en, self.pr,self.vecen,self.vecpr)['out'].cpu().numpy()
        # auc_val = roc_auc_score(self.y.cpu().numpy(), y_pred)
        # aupr_val = average_precision_score(self.y.cpu().numpy(), y_pred)
        
        self.model.eval()
        y_true_all = []
        y_pred_all = []
        with torch.no_grad():
            for x_en, x_pr, x_en_vec, x_pr_vec, y in self.val_loader:
                x_en = x_en.to(self.device)
                x_pr = x_pr.to(self.device)
                x_en_vec = x_en_vec.to(self.device)
                x_pr_vec = x_pr_vec.to(self.device)

                out = self.model(x_en, x_pr, x_en_vec, x_pr_vec)['out']
                y_pred_all.extend(out.cpu().numpy())
                y_true_all.extend(y.cpu().numpy())

        auc_val = roc_auc_score(y_true_all, y_pred_all)
        aupr_val = average_precision_score(y_true_all, y_pred_all)

        # 保存模型
        torch.save(self.model.state_dict(), f"{self.save_path}{self.name}Model{epoch}_xr2.pth")

        print(f"\rAUC: {round(auc_val, 4)}", end="\n")
        print(f"\rAUPR: {round(aupr_val, 4)}", end="\n")


def compute_avg_row_frequencies(X, save_path='X_avg_freq.npz'):
    """
    计算每个数字在每一行的出现频率，求和后除以3000，得到整体频率向量。

    参数：
        X (np.ndarray): 输入数组，形状 (N, 3000)
        save_path (str): 保存路径
    """
    
    # 如果文件已存在，则跳过计算
    if os.path.exists(save_path):
        print(f"文件 {save_path} 已存在，跳过计算。")
        return
    assert isinstance(X, np.ndarray), "X 必须是 np.ndarray"
    N, D = X.shape
    assert D == 3000 or 2000, "每行应包含 3000 个元素"
    
    max_val = 4096
    freq_accum = np.zeros((N, max_val + 1), dtype=np.float64)

    for i in range(N):
        row = X[i]
        row_counts = np.bincount(row, minlength=max_val + 1) / D  # 每行频率
        freq_accum[i] = row_counts

    avg_freq = freq_accum   # 最终频率向量

    np.savez_compressed(save_path, frequencies=avg_freq)
    print(f"频率向量已保存至：{save_path}，shape={avg_freq.shape}")
        
def evaluate_model(model, data_loader, device):
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for x_en, x_pr, x_en_vec, x_pr_vec, y in data_loader:
            x_en, x_pr, x_en_vec, x_pr_vec = x_en.to(device), x_pr.to(device), x_en_vec.to(device), x_pr_vec.to(device)
            y_pred = model(x_en, x_pr, x_en_vec, x_pr_vec)['out'].cpu().numpy()
            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend(y_pred)
    auc = roc_auc_score(y_true_all, y_pred_all)
    aupr = average_precision_score(y_true_all, y_pred_all)
    return auc, aupr


# 获取当前时间
t1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 细胞系名称列表
names = ['GM12878', 'HUVEC', 'HeLa', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
name = names[5]

# 数据路径和加载
Data_dir = f'../EPIVAN-master/Data/rawData/{name}/'
Data_dirnpz = f'../EPIVAN-master/Data/'
train = np.load(Data_dir + f'{name}_train.npz')
test = np.load(Data_dir + f'{name}_test.npz')
X_en_tra, X_pr_tra, y_tra = train['X_en_tra'], train['X_pr_tra'], train['y_tra']
X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']

compute_avg_row_frequencies(X_en_tra,Data_dirnpz+f'X_en_tra_{name}.npz')
compute_avg_row_frequencies(X_pr_tra,Data_dirnpz+f'X_pr_tra_{name}.npz')
compute_avg_row_frequencies(X_en_tes,Data_dirnpz+f'X_en_tes_{name}.npz')
compute_avg_row_frequencies(X_pr_tes,Data_dirnpz+f'X_pr_tes_{name}.npz')


# # 随机采样
# num_train_samples = 3000
# num_test_samples = 1000
# train_indices = np.random.choice(len(X_en_tra), num_train_samples, replace=False)
# test_indices = np.random.choice(len(X_en_tes), num_test_samples, replace=False)

# X_en_tra, X_pr_tra, y_tra = X_en_tra[train_indices], X_pr_tra[train_indices], y_tra[train_indices]
# X_en_tes, X_pr_tes, y_tes = X_en_tes[test_indices], X_pr_tes[test_indices], y_tes[test_indices]


#读取频率序列
vector_en_tra = np.load(f'X_en_tra_{name}.npz')['frequencies']
vector_pr_tra = np.load(f'X_pr_tra_{name}.npz')['frequencies']
vector_en_tes = np.load(f'X_en_tes_{name}.npz')['frequencies']
vector_pr_tes = np.load(f'X_pr_tes_{name}.npz')['frequencies']
# vector_en_tra,vector_pr_tra=vector_en_tra[train_indices],vector_pr_tra[train_indices]
# vector_en_tes,vector_pr_tes=vector_en_tes[test_indices],vector_pr_tes[test_indices]
# print(f"Training vector_en shape: {vector_en_tra.shape}")
# print(f"Training vector_pr shape: {vector_pr_tra.shape}")
# print(f"Testing vector_en shape: {vector_en_tes.shape}")
# print(f"Testing vector_pr shape: {vector_pr_tes.shape}")


# 划分训练集和验证集
X_en_tra, X_en_val, X_pr_tra, X_pr_val, y_tra, y_val,vector_en_tra, vector_en_val, vector_pr_tra, vector_pr_val = train_test_split(
    X_en_tra, X_pr_tra, y_tra,vector_en_tra, vector_pr_tra, test_size=0.1, stratify=y_tra, random_state=250)




# 转换为 PyTorch tensor
X_en_tra = torch.tensor(X_en_tra, dtype=torch.long)
X_pr_tra = torch.tensor(X_pr_tra, dtype=torch.long)
y_tra = torch.tensor(y_tra, dtype=torch.float32).view(-1, 1)


X_en_val = torch.tensor(X_en_val, dtype=torch.long)
X_pr_val = torch.tensor(X_pr_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_en_tes = torch.tensor(X_en_tes, dtype=torch.long)
X_pr_tes = torch.tensor(X_pr_tes, dtype=torch.long)
y_tes = torch.tensor(y_tes, dtype=torch.float32).view(-1, 1)

vector_en_tra=torch.tensor(vector_en_tra, dtype=torch.float32)
vector_pr_tra=torch.tensor(vector_pr_tra, dtype=torch.float32)

vector_en_val=torch.tensor(vector_en_val, dtype=torch.float32)
vector_pr_val=torch.tensor(vector_pr_val, dtype=torch.float32)

vector_en_tes=torch.tensor(vector_en_tes, dtype=torch.float32)
vector_pr_tes=torch.tensor(vector_pr_tes, dtype=torch.float32)




# 加载嵌入矩阵
embedding_matrix = np.load('embedding_matrix.npy')

# 初始化模型
model = GenomicModel(
    embedding_matrix=embedding_matrix,
    max_len_en=3000,
    max_len_pr=2000,
    nb_words=4097,
    embedding_dim=100,
    num_heads=5,
)

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# 创建 DataLoader
train_dataset = TensorDataset(X_en_tra, X_pr_tra,vector_en_tra, vector_pr_tra, y_tra)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(X_en_val, X_pr_val,vector_en_val, vector_pr_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

tes_dataset = TensorDataset(X_en_tes, X_pr_tes, vector_en_tes, vector_pr_tes, y_tes)
tes_loader = DataLoader(tes_dataset, batch_size=16, shuffle=False)


tra_data=(
        # X_en_tra.to(device),
        # X_pr_tra.to(device),
        # vector_en_tra.to(device),
        # vector_pr_tra.to(device),
        # y_tra.to(device)
        X_en_tra,
        X_pr_tra,
        vector_en_tra,
        vector_pr_tra,
        y_tra
    )

tes_data=(
        X_en_tes.to(device),
        X_pr_tes.to(device),
        vector_en_tes.to(device),
        vector_pr_tes.to(device),
        y_tes.to(device)
    )

val_data=(
        X_en_val.to(device),
        X_pr_val.to(device),
        vector_en_val.to(device),
        vector_pr_val.to(device),
        y_val.to(device)
    )


# 自定义回调
roc_callback = RocCallback(
    val_loader=val_loader,
    name=name,
    model=model,
    device=device
)

# 训练过程
print(f"Training {name} cell line specific model ...")
num_epochs = 30

# c_weight = 0



for epoch in range(num_epochs):
    model.train()
    running_loss = 0.05
    #对比学习损失
    running_classification_loss = 0.0
    running_contrastive_loss = 0.0
    for X_en_batch, X_pr_batch,X_en_vec_batch,X_pr_vec_batch, y_batch in train_loader:
        # 将数据移动到 GPU
        X_en_batch, X_pr_batch, y_batch,X_en_vec_batch,X_pr_vec_batch = X_en_batch.to(device), X_pr_batch.to(device), y_batch.to(device),X_en_vec_batch.to(device),X_pr_vec_batch.to(device)

        # 前向传播和计算损失
        optimizer.zero_grad()
        output = model(X_en_batch, X_pr_batch,X_en_vec_batch,X_pr_vec_batch)
        
        
        loss_e = output['loss_e']
        loss_p = output['loss_p']
        out = output['out']
        
        loss = criterion(out, y_batch)
        # loss = loss + c_weight * (loss_e + loss_p)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    
    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(tra_data[0], tra_data[1],tra_data[2],tra_data[3])['out'].cpu().numpy()
    # auc_tra = roc_auc_score(tra_data[4].cpu().numpy(), y_pred)
    # aupr_tra = average_precision_score(tra_data[4].cpu().numpy(), y_pred)
    # print(f"\rtra AUC: {round(auc_tra, 4)}", end="\n")
    # print(f"\rtra AUPR: {round(aupr_tra, 4)}", end="\n")

    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(val_data[0], val_data[1],val_data[2],val_data[3])['out'].cpu().numpy()
    # auc_val = roc_auc_score(val_data[4].cpu().numpy(), y_pred)
    # aupr_val = average_precision_score(val_data[4].cpu().numpy(), y_pred)
    auc_val, aupr_val = evaluate_model(model, val_loader, device)


    print(f"\rval AUC: {round(auc_val, 4)}", end="\n")
    print(f"\rval AUPR: {round(aupr_val, 4)}", end="\n")

    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(tes_data[0], tes_data[1],tes_data[2],tes_data[3])['out'].cpu().numpy()
    # auc_tes = roc_auc_score(tes_data[4].cpu().numpy(), y_pred)
    # aupr_tes = average_precision_score(tes_data[4].cpu().numpy(), y_pred)
    
    auc_tes, aupr_tes = evaluate_model(model, tes_loader, device)


    print(f"\rtes AUC: {round(auc_tes, 4)}", end="\n")
    print(f"\rtes AUPR: {round(aupr_tes, 4)}", end="\n")

    # 在每个 epoch 结束时调用自定义回调
    roc_callback.on_epoch_end(epoch)

# 结束时间
t2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"开始时间: {t1}, 结束时间: {t2}")
