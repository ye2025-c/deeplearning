import random 
import torch
import d2l.torch as d2l

'''我们将使用低维数据，这样可以很容易地将其可视化。在下面的代码中，我们
生成一个包含1000个样本的数据集，每个样本包含从标准正态分布中采样的2个特征。我们的合成数据集是
一个矩阵X∈R1000×2。
'''
# 生成数据集
def synthetic_data(w,b,num_examples):
    X = torch.normal(0,1,(num_examples,len(w)))       #从均值为0、标准差为1的正态分布中随机采样
    y = torch.matmul(X,w) + b                 #matmul是矩阵乘法
    y += torch.normal(0,0.01,y.shape)         #在标签上添加噪声，噪声的形状与y相同，均值为0，标准差为0.01
    return X,y.reshape((-1,1))

#设定真实的权重和偏差
true_w = torch.tensor([2,-3.4])
true_b = 4.2

features,labels = synthetic_data(true_w,true_b,1000)

print('features:',features[0],"\nlabel:",labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1)  #以第二权重对应的特征为横坐标，标签为纵坐标，绘制散点图，点的大小为1
d2l.plt.show()

# 读取数据
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]


# 初始化模型参数
w = torch.normal(0,0.001,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)


# 定义模型
def linreg(X,w,b):
    return torch.matmul(X,w) + b

# 定义损失函数
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

# 定义优化算法
'''小批量随机梯度下降'''
def sgd(params,lr,batch_size):
    with torch.no_grad():   #不需要存入计算图，所以可以防止发生梯度更新
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

'''训练模型'''
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y) #X和y的小批量损失
        #计算[w,b]的梯度
        l.sum().backward()
        sgd([w,b],lr,batch_size)  #更新参数
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
