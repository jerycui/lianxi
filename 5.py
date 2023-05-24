import torch
import numpy as np
# 定义输入特征和隐藏层特征的维度
input_size = 3
hidden_size = 4

# 定义RNN的权重参数
Wxh = torch.randn(hidden_size, input_size)  # 输入层到隐藏层的权重矩阵
Whh = torch.randn(hidden_size, hidden_size)  # 隐藏层到隐藏层的权重矩阵
Why = torch.randn(2, hidden_size)  # 隐藏层到输出层的权重矩阵
bh = torch.zeros(hidden_size, 1)  # 隐藏层的偏置向量
by = torch.zeros(2, 1)  # 输出层的偏置向量

# 定义激活函数
def softmax(x):
    return np.exp(x)/np.exp(x).sum()

# 定义RNN的前向传播函数
def rnn_forward(x, h_prev):
    # 计算隐藏层状态
    h = softmax(torch.mm(Wxh, x) + torch.mm(Whh, h_prev) + bh)
    # 计算输出层状态
    y = torch.mm(Why, h) + by
    # 返回隐藏层状态和输出层状态
    return h, y

# 定义输入数据和初始隐藏层状态
x = torch.tensor([[0.1], [0.2], [0.3]])  # 输入数据，shape为(3, 1)
h_prev = torch.tensor([[0.4], [0.5], [0.6], [0.7]])  # 初始隐藏层状态，shape为(4, 1)

# 调用RNN前向传播函数进行计算
h, y = rnn_forward(x, h_prev)

# 打印输出层状态和隐藏层状态
print("Output:", y)
print("Hidden state:", h)


# import torch
# import torch.nn as nn
#
#
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#
#         # 定义RNN中用到的三个参数
#         self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size))
#         self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
#         self.Why = nn.Parameter(torch.randn(hidden_size, output_size))
#
#         # 定义激活函数
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, input, hidden):
#         # 计算隐层状态
#         hidden = self.tanh(torch.mm(input, self.Wxh) + torch.mm(hidden, self.Whh))
#         # 计算输出
#         output = self.softmax(torch.mm(hidden, self.Why))
#         # 返回输出和隐层状态
#         return output, hidden
#
#     def init_hidden(self):
#         return torch.zeros(1, self.hidden_size)
#
#
# # 定义输入，隐层，输出维度
# input_size = 16
# hidden_size = 8
# output_size = 6
#
# # 定义模型
# rnn = RNN(input_size, hidden_size, output_size)
#
# # 测试模型
# input = torch.randn(1, input_size)
# # hidden = rnn.init_hidden()
# hidden = torch.zeros(1,hidden_size)
# output, hidden = rnn(input, hidden)
# print(output.detach().numpy())
# print(hidden.detach().numpy())




