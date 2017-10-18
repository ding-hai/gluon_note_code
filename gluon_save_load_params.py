#coding：utf8
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from utils import load_data_fashion_mnist,accuracy,evaluate_accuracy

import os


#超参数
batch_size = 10
num_inputs = 28*28
num_hiddens = 256
num_outputs = 10
learning_rate = .01

#加载数据
train_data,test_data = load_data_fashion_mnist(batch_size)

#定义模型
class MLP(gluon.nn.Block):
    def __init__(self,**kwargs):
        super(MLP,self).__init__(**kwargs)
        with self.name_scope():
            #指定每一层的in_units就不会延迟到第一次接触数据的时候初始化参数
            self.input = gluon.nn.Dense(num_inputs,in_units=num_inputs)
            self.hidden_1 = gluon.nn.Dense(num_hiddens,in_units=num_inputs,activation="relu")
            self.output = gluon.nn.Dense(num_outputs,in_units=num_hiddens)

    def forward(self,X):
        return self.output(self.hidden_1(self.input(X)))


net = MLP()
net.initialize()

#损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


#训练
trainer = gluon.Trainer(net.collect_params(),'sgd',{"learning_rate":learning_rate})
if os.path.isfile("./net_params.params"):
    net.load_params(net_params.params,mx.cpu())
    test_acc = evaluate_accuracy(test_data,net)
    print("测试精度",test_acc)
else:
    for epoch in range(5):
        train_loss = .0
        train_acc = .0
        num_batchs = len(train_data)
        for data,label in train_data:
            with ag.record():
                output = net(data)
                loss = softmax_cross_entropy(output,label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output,label)
        test_acc = evaluate_accuracy(test_data,net)
        print("Epoch:%d train_loss:%f, train_acc:%f, test_acc:%f"%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))
    net.save_params("./net_params.params")
