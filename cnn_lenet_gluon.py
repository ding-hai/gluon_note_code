#coding：utf8
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from utils import load_data_fashion_mnist,accuracy,Result
import mxnet as mx
import json
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


def evaluate_accuracy(data_iterator,net):
    total_acc =0.0
    for data,label in data_iterator:
        data = data.reshape((-1,1,28,28))
        output = net(data)
        total_acc+=accuracy(output,label)
    return total_acc/len(data_iterator)

#搭建模型
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(
    gluon.nn.Conv2D(channels=20,kernel_size=5,activation='relu'),
    gluon.nn.MaxPool2D(pool_size=2,strides=2),
    gluon.nn.Conv2D(channels=50,kernel_size=3,activation='relu'),
    gluon.nn.MaxPool2D(pool_size=2,strides=2),
    gluon.nn.Flatten(),
    gluon.nn.Dense(128,activation="relu"),
    gluon.nn.Dense(10)
    )
net.initialize()

#损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{"learning_rate":learning_rate})
# 训练
result = Result()

for epoch in range(100):
    train_loss = .0
    train_acc = .0
    num_batchs = len(train_data)
    for data,label in train_data:
        with ag.record():
            data=data.reshape((-1,1,28,28))
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net)
    result.add("train_loss",train_loss/num_batchs)
    result.add("train_acc",train_acc/num_batchs)
    result.add("test_acc",test_acc)
    print("Epoch:%d train_loss:%f train_acc:%f, test_acc:%f"%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))

with open("lenet_result_map.json",'w') as file:
    json.dump(result.result_map,file)
