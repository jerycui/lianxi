import torch
import torch.nn as nn
import pandas as pd
import gensim
import logging
# 定义模型
# class RNNModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim):
#         super(RNNModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(2 * hidden_dim, 1)
#
#     def forward(self, x1, x2):
#         x1_embedded = self.embedding(x1)
#         x2_embedded = self.embedding(x2)
#         _, x1_hidden = self.rnn(x1_embedded)
#         _, x2_hidden = self.rnn(x2_embedded)
#         x = torch.cat((x1_hidden[-1], x2_hidden[-1]), dim=1)
#         x = self.fc(x)
#         return x.squeeze()
#
#
# # 定义数据、标签和模型
# x1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# x2 = torch.tensor([[1, 2, 3], [7, 8, 9]])
# label = torch.tensor([1, 0])
# model = RNNModel(vocab_size=10, embedding_dim=50, hidden_dim=100)
#
# # 定义损失函数和优化器
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters())
# num_epochs = 10000
# # 训练模型
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = model(x1, x2)
#     loss = criterion(outputs, label.float())
#     # 反向传播
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     # 打印损失
#     # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
#     print('Epoch [{}/{}],loss:{}'.format(epoch+1,num_epochs,loss.item()))
#
# print(model(x1,x1).detach().numpy())
# print(model(x1,x2).detach().numpy())

train = pd.read_csv('contradictory-my-dear-watson/train.csv')
# print(train.head())
import seaborn as sns
import matplotlib.pyplot as plt

# sns.displot(data.label)
# plt.show()
# print(train['language'].value_counts())


def count_values_by_category(df, count_column_name, category_column_name):
    """
    Receives a pandas dataframe, a column with categorical values and a column
    to be counted, and returns a new dataframe with the count of occurrences of
    each count_column value for each category_column value.
    """
    # Group the DataFrame by the category_column and count the values in the count_column
    counts_df = df.groupby(category_column_name)[count_column_name].value_counts().unstack(fill_value=0)

    return counts_df


counts_df = count_values_by_category(train, 'label', 'lang_abv')
print(counts_df)