import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的RNN模型


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


# 超参数
input_size = 10
hidden_size = 20
output_size = 1
learning_rate = 0.001

# 实例化模型、损失函数和优化器
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 示例输入和目标数据
# (batch_size, seq_length, input_size)
input_data = torch.randn(1, 5, input_size)
# (batch_size, seq_length, output_size)
target_data = torch.randn(1, 5, output_size)

# 训练循环
for epoch in range(100):
    hidden = torch.zeros(1, 1, hidden_size)  # 初始化隐藏状态
    optimizer.zero_grad()  # 梯度清零

    output, hidden = model(input_data, hidden)  # 前向传播
    loss = criterion(output, target_data)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")
