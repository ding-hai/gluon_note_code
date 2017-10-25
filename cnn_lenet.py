#coding：utf8
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from utils import load_data_fashion_mnist,accuracy,evaluate_accuracy,SGD
import mxnet as mx
ctx =mx.cpu()

#超参数
batch_size = 256
num_inputs = 28*28
num_hiddens = 256
num_outputs = 10
learning_rate = .2
weight_scale = .01
#加载数据
train_data,test_data = load_data_fashion_mnist(batch_size)


#搭建模型
## 模型参数
### 第一个卷积层输入是一个通道输出有20个通道，也就是说有20个卷积核 5x5
w1 = nd.random_normal(shape=(20,1,5,5),scale=weight_scale)
b1 = nd.zeros(w1.shape[0])

### 第二个卷积层输入是第一个的输出：20个通道，输出时50个通道，意味着有20x50个卷积核 3x3
w2 = nd.random_normal(shape=(50,20,3,3),scale=weight_scale)
b2 = nd.zeros(w2.shape[0])

### 全连接层
w3 = nd.random_normal(shape=(1250,128),scale=weight_scale)
b3 = nd.zeros(w3.shape[1])

w4 = nd.random_normal(shape=(128,10),scale=weight_scale)
b4 = nd.zeros(w4.shape[1])

params = [w1, b1,w2,b2,w3,b3,w4,b4]
for param in params:
    param.attach_grad()


def net(data,verbose=False):
    X=data.reshape((-1,1,28,28))
    h1_conv = nd.Convolution(data=X,weight=w1,bias=b1,kernel=w1.shape[2:],num_filter=w1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1_pool = nd.Pooling(data=h1_activation,pool_type="max",kernel=(2,2),stride=(2,2))

    h2_conv = nd.Convolution(data=h1_pool,weight=w2,bias=b2,kernel=w2.shape[2:],num_filter=w2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2_pool = nd.Pooling(data=h2_activation,pool_type="max",kernel=(2,2),stride=(2,2))

    h2 = nd.flatten(h2_pool)

    #全连接
    h3_linar = nd.dot(h2,w3)+b3
    h3_activation = nd.relu(h3_linar)

    h4_linar = nd.dot(h3_activation,w4)+b4
    #h4_activation = nd.relu(h4_linar)

    if verbose:
        print("conv_1 out shape:",h1_pool.shape)
        print("conv_2 out shape:",h2_pool.shape)
        print("full conect 1 out shape:",h3_activation.shape)
        print("full connect 2 out shape:",h4_linar.shape)
    return h4_linar#h4_activation


#损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
# 训练
for epoch in range(100):
    train_loss = .0
    train_acc = .0
    for data,label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        SGD(params,learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net)
    print("Epoch:%d train_loss:%f train_acc:%f, test_acc:%f"%(epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))
