import numpy as np
import tushare as ts
import torch
from torch import nn

# 获取上证指数从20180101开始的收盘价的np.ndarray
data_close = ts.get_k_data('000001', start='2018-01-01', index=True)['close'].values
data_close = data_close.astype('float32')  # 转换数据类型
print(data_close)  # 这个数据为一组数字序列：[1234.98， 7483.09， 3091.47,...]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 将价格标准化到0~1
# 1.标准化：
# 　　数据标准化是指数据的各维度减均值除以标准差，这是最常用的标准化方法。
# 　　公式：(xi−μ)/σ 其中μ指的是样本的均值，σ指的是样本的标准差。
#
# 2.归一化：
# 　　数据归一化是指数据减去对应维度的最小值除以维度最大值减去维度最小值，这样做可以将数值压缩到[0,1]的区间。
# 　　公式：(xi−min(x))/(max(x)−min(x))
#  意义：根据反向传播公式。如果输入层 x 很大，在反向传播时候传递到输入层的梯度就会变得很大。
#  梯度大，学习率就得非常小，否则会越过最优。
#  在这种情况下，学习率的选择需要参考输入层数值大小，而直接将数据归一化操作，能很方便的选择学习率。

max_value = np.max(data_close)  # numpy的方法，分别获取数据中的最大值和最小值
min_value = np.min(data_close)
data_close = (data_close - min_value) / (max_value - min_value)

# 那么这个DAYS_FOR_TRAIN的意义是什么呢？我们获取到的K线股票数据，本质上是一系列不同天数对应的股票指数的变化的数据
# 那么通过分割，每DAYS_FOR_TRAIN个收盘价对应一个未来的收盘价。例如K线为 [1,2,3,4,5]， DAYS_FOR_TRAIN=3，那么将会生成如下2组数据：
# 第1组的输入是 [1,2,3]，对应输出 4；
# 第2组的输入是 [2,3,4]，对应输出 5。通过这样分割，我们也就实现了对数据的x输入和y输出的标定。
DAYS_FOR_TRAIN = 10


def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集

        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。

        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y = [], []  # 在这里定义两个列表,dataset_x存放若干个长度为（days_for_train)的序列,dateset_y则存放相应序列对应的那一个数据。
    for i in range(len(data) - days_for_train):  # 开始对传入的序列data进行迭代，迭代次数为序列的长度-days_for_train。
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return np.array(dataset_x), np.array(dataset_y)  # 最后将两个列表返回出去


dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)
print(dataset_x)
print(dataset_y)

# 划分训练集和测试集，70%作为训练集。
train_size = int(len(dataset_x) * 0.7)

train_x = dataset_x[:train_size]
train_y = dataset_y[:train_size]

# 将数据改变形状，RNN(LSTM) 读入的数据维度由三个参数组成 (seq_len, batch_size, input_dim)
# 比如我们输入一个文本：
# seq_len是一个句子的最大长度，比如15,则代表最长有15单词。
# batch_size是一次往RNN输入句子的数目，比如是5。则一次输入的句子数目为5.
# input_dim是输入的维度，比如是128，则每个单词用长度为128的的向量表示。
# 可以理解为现在一共有batch_size个独立的RNN组件，RNN的每一个输入维度是input_dim，每次总共输入seq_len个时间步
# 对于我们这次使用的数据：[[2340 7620 8331 9546 1235...],[1345 6537 2384 9012 2512...]...]
# 在这里，我们是所有序列作为一个句子输入进去
train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
train_y = train_y.reshape(-1, 1, 1)

# 转为pytorch的tensor对象
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)


# 神经网络的典型处理如下：
# 1. 定义可学习参数的网络结构（堆叠各层和层的设计）；
# 2. 数据集输入；
# 3. 对输入进行处理（由定义的网络层进行处理）,主要体现在网络的前向传播；
# 4. 计算loss ，由Loss层计算；
# 5. 反向传播求梯度；
# 6. 根据梯度改变参数值


class LSTM_Regression(nn.Module):  # 继承了nn.Module，继承后需要实现forward方法，并在构造函数中调用module的构造函数
    """
        使用LSTM进行回归

        参数：
        - input_size: feature size。输入特征的维数，前面定义的input dim。
        - hidden_size: number of hidden units。隐藏层状态的维数，即隐藏层节点的个数，这个和单层感知器的结构是类似的。
        - output_size: number of output。输出特征的维数
        - num_layers: layers of LSTM to stack。LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果。
    """

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):  # 一般把网络中具有可学习参数的层放在构造函数中
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # 定义了LSTM网络层的相关参数
        # nn.Linear这个函数的功能是用于设置网络中的全连接层。这里面的fc就是全连接层。第一个参数指输入的二维张量的大小，第二个参数指出输出的二维张量的大小。在这里我们是作为最后一层。
        # 从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
        # 这样做，实现了最后一个隐藏层到输出层的衔接
        self.fc = nn.Linear(hidden_size, output_size)

    # 先看最下面
    # model(x)会调用基类的_call_方法，_call_再调用forward方法。
    def forward(self, _x):  # 只要在nn.module中定义了forward前向函数, backward后向函数就会被自动实现
        # _x直接扔进去。 _x is input, size (seq_len, batch, input_size) 句子长度，句子数目，单词长度
        x, _ = self.lstm(_x)  # 输出(hn, cn) 最后时刻的隐层状态和记忆单元中的值
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size) 句子长度，句子数目，隐藏单元数目
        # 因为全连接层只接受二维张量，因此调用view方法进行变形
        x = x.view(s * b, h)
        x = self.fc(x)  # 扔进去，输出一个out_features为1的二维张量 (s * b, 1)
        x = x.view(s, b, -1)  # 把形状改回来，参数为-1，则在前两个参数已知的情况下自动补齐向量长度。
        # x代表每个句子的每个单词的最终一维输出——组成的三维张量
        return x


# lstm中，输入特征维数为DAYS_FOR_TRAIN，可以理解成单词的长度。隐藏层节点为8，输出特征维数为1，共2层LSTM
model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)
loss_function = nn.MSELoss()  # 定义损失函数，是均方损失函数: (xi-yi)平方
loss_function = loss_function.cuda()

# 在这里，我们构造一个优化器对象optimizer，来保存当前状态，并根据计算得到的梯度来更新参数
# torch.optim是一个实现了多种优化算法的包。调用Adam算法。
# Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# params：待优化参数的迭代器或dict，lr：学习率，betas：用于计算梯度以及梯度平方的运行平均值的系数，eps：为了增加数值计算的稳定性而加到分母的值，weigh_decay：权重衰减
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for i in range(1000):   # 训练1000个epoch，即把全部数据训练1000次
    train_x, train_y = train_x.cuda(), train_y.cuda()
    out = model(train_x)  # 自动调用forward()方法
    loss = loss_function(out, train_y)
    loss.backward()  # 反向传播求梯度

    # 调用step()方法，会更新所有的参数
    optimizer.step()
    # zero_grad用于每次计算完一个batch样本后的梯度清零
    # why：pytorch中梯度反馈在节点上是累加的，每一个batch时不需要将两个batch的梯度混合起来累积，因此这里每个batch设置一次zero_grad
    optimizer.zero_grad()

    if (i + 1) % 100 == 0:
        print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))


import matplotlib.pyplot as plt

model = model.eval()  # 转换成测试模式

# 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
dataset_x = torch.from_numpy(dataset_x)
dataset_x = dataset_x.cuda()

pred_test = model(dataset_x)  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
pred_test = pred_test.cpu()
pred_test = pred_test.view(-1).data.numpy()  # 转化为一维后转成numpy
pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))  # 填充0 使长度相同
assert len(pred_test) == len(data_close)    # 确认长度相同则通过

plt.plot(pred_test, 'r', label='prediction')
plt.plot(data_close, 'b', label='real')
plt.plot((train_size, train_size), (0, 1), 'g--')
plt.legend(loc='best')
plt.show()
