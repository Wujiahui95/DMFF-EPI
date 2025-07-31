import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义多头注意力层（Multi-Head Attention Layer）
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, attention_dim, num_heads):
        """
        初始化多头注意力层。
        :param attention_dim: 注意力机制的隐藏层维度
        :param num_heads: 多头注意力头的数量
        """
        super(MultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value, mask=None):
        """
        前向传播函数。
        :param query: 查询张量，形状为 [batch_size, seq_len, attention_dim]
        :param key: 键张量，形状为 [batch_size, seq_len, attention_dim]
        :param value: 值张量，形状为 [batch_size, seq_len, attention_dim]
        :param mask: 可选的掩码张量
        :return: 通过多头注意力计算得到的输出
        """
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=mask)
        return attn_output

    

#对比学习框架    
class SimCLRRegularizer(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRRegularizer, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        输入：
        z1: [B, D] - 第一种模态输出（如 CNN 表示）
        z2: [B, D] - 第二种模态输出（如 Transformer 表示）
        返回：
        loss: scalar 对比损失
        """
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)  # [2B, D]
        similarity_matrix = torch.matmul(representations, representations.T)  # [2B, 2B]
        similarity_matrix = similarity_matrix / self.temperature

        labels = torch.cat([torch.arange(batch_size) + batch_size,
                            torch.arange(batch_size)], dim=0).to(z1.device)  # 正样本索引
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    
# 定义主模型
class GenomicModel(nn.Module):
    def __init__(self, embedding_matrix, max_len_en, max_len_pr, nb_words, embedding_dim, num_heads,temperature=0.5):
        """
        初始化基因组模型。
        :param embedding_matrix: 预训练的嵌入矩阵
        :param max_len_en: 增强子序列的最大长度
        :param max_len_pr: 启动子序列的最大长度
        :param nb_words: 词汇表大小
        :param embedding_dim: 嵌入维度
        :param num_heads: 多头注意力头的数量
        :param temperature: 对比学习温度参数
        """
        super(GenomicModel, self).__init__()

        # 定义嵌入层，加载预训练嵌入矩阵
        self.emb_en = nn.Embedding(nb_words, embedding_dim)
        self.emb_pr = nn.Embedding(nb_words, embedding_dim)
        self.emb_en.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.emb_pr.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.emb_en.weight.requires_grad = True
        self.emb_pr.weight.requires_grad = True

        # 定义卷积层
        self.conv_en = nn.Conv1d(embedding_dim, 64, kernel_size=40, padding=0)
        self.conv_pr = nn.Conv1d(embedding_dim, 64, kernel_size=40, padding=0)

        # 定义最大池化层
        self.pool = nn.MaxPool1d(kernel_size=20, stride=20)

        # 定义批归一化和 Dropout
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # 定义双向 GRU 层
        self.gru1 = nn.GRU(64, 50, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(64, 50, bidirectional=True, batch_first=True)
        
        #定义mlp层
        self.mlp1 = nn.Sequential(
                nn.Linear(4097, 256),  # 第一层
                nn.ReLU(),
                nn.Dropout(p=0.5),   # 第一层后加入 Dropout
                nn.Linear(256, 100)    # 第二层
            )
        self.mlp2 = nn.Sequential(
                nn.Linear(4097, 256),  # 第一层
                nn.ReLU(),
                nn.Dropout(p=0.5),   # 第一层后加入 Dropout
                nn.Linear(256, 100)    # 第二层
            )

        # 定义多头注意力层
        # self.multihead_attention = MultiHeadAttentionLayer(100, num_heads)
        
        self.Regularizer_e = SimCLRRegularizer(temperature)
        self.Regularizer_p = SimCLRRegularizer(temperature)

        # 定义全连接层
        self.fc = nn.Linear(400, 1)

    def forward(self, enhancers, promoters,vector_en,vector_pr):
        """
        前向传播函数。
        :param enhancers: 增强子输入，形状为 [batch_size, max_len_en]
        :param promoters: 启动子输入，形状为 [batch_size, max_len_pr]
        :return: 输出的分类概率，形状为 [batch_size, 1]
        """
        # 嵌入层
        emb_en = self.emb_en(enhancers).permute(0, 2, 1)  # [batch_size, embedding_dim, max_len_en]
        emb_pr = self.emb_pr(promoters).permute(0, 2, 1)  # [batch_size, embedding_dim, max_len_pr]

        # 卷积和池化
        conv_en = self.pool(F.relu(self.conv_en(emb_en)))  # [batch_size, 64, seq_len_en]
        conv_pr = self.pool(F.relu(self.conv_pr(emb_pr)))  # [batch_size, 64, seq_len_pr]

        # 拼接增强子和启动子特征
        merged = torch.cat((conv_en, conv_pr), dim=2)  # [batch_size, 64, seq_len]

        # 批归一化和 Dropout
        # merged = self.bn(merged)
        # merged = self.dropout(merged.permute(0, 2, 1))  # [batch_size, seq_len, 64]
        merged_en = self.bn1(conv_en)
        merged_pr = self.bn2(conv_pr)
        merged_en = self.dropout1(merged_en.permute(0, 2, 1))  # [batch_size, seq_len, 64]
        merged_pr = self.dropout1(merged_pr.permute(0, 2, 1))  # [batch_size, seq_len, 64]
        
        # 双向 GRU
        gru_out_en, _ = self.gru1(merged_en)  # [batch_size, seq_len, 100]
        gru_out_pr, _ = self.gru2(merged_pr)  # [batch_size, seq_len, 100]
        
        gru_out_en=gru_out_en.mean(1)
        gru_out_pr=gru_out_pr.mean(1)
#         # Cross-Attention: Enhance promoters with enhancer context and vice versa
#         cross_attn_en_to_pr = self.multihead_attention(gru_out, gru_out, gru_out)  # [batch_size, seq_len, 100]
#         cross_attn_pr_to_en = self.multihead_attention(gru_out, gru_out, gru_out)  # [batch_size, seq_len, 100]
#         cross_attn_output = (cross_attn_en_to_pr + cross_attn_pr_to_en) / 2  # 合并两种交叉注意力


        x_en_m=self.mlp1(vector_en)              #vector_en   batch_size, 4096
        x_pr_t=self.mlp2(vector_pr)              #vector_pr   batch_size, 4096

        loss_e = self.Regularizer_e(x_en_m, gru_out_en)
        loss_p = self.Regularizer_p(x_pr_t, gru_out_pr)

        # 全连接层
        # output = self.fc(cross_attn_output.mean(dim=1))  # [batch_size, 1]
        out = self.fc(torch.cat([gru_out_en, x_en_m, gru_out_pr, x_pr_t], dim=1))  # [batch_size, 1]
        # out = self.fc(torch.cat([gru_out_en, x_en_m], dim=1))  # [batch_size, 1]
        out = torch.sigmoid(out)  # 输出分类概率
        output = dict()
        output['out'] = out
        output['loss_e'] = loss_e
        output['loss_p'] = loss_p
        return output 
    