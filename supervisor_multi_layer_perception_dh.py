#coding:utf8
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon
from utils import load_data_fashion_mnist,accuracy,evaluate_accuracy,SGD,Result


#超参数
batch_size = 10
num_inputs = 28*28
num_hiddens = 256
num_outputs = 10
learning_rate = .01
#加载数据
train_data ,test_data = load_data_fashion_mnist(batch_size)

#定义模型
## 一个28*28的输入层，一个256的隐含层，一个10的输出层
W1 = nd.random_normal(shape=(num_inputs,num_hiddens))
b1 = nd.random_normal(shape=(num_hiddens))

W2 = nd.random_normal(shape=(num_hiddens,num_outputs))
b2 = nd.random_normal(shape=(num_outputs))


params = [W1,b1,W2,b2]
for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X,0)

def net(X):
    temp = relu(nd.dot(X.reshape((-1,num_inputs)),W1)+b1)
    output = nd.dot(temp,W2)+b2
    return output

# 损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#训练
result = Result()
for epoch in range(20):
    train_loss = .0
    train_acc = .0
    num_batchs = len(train_data)
    for data,label in train_data:
        with ag.record() :
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        SGD(params,learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net)
    result.add("train_loss",train_loss/num_batchs)
    result.add("train_acc",train_acc/num_batchs)
    result.add("test_acc",test_acc)

    print("Epoch %d: train_loss :%f, train_acc:%f, test_acc:%f"%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))
result.show()
