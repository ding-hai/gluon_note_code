#coding:utf8
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon
from utils import load_data_fashion_mnist,accuracy,evaluate_accuracy,Result


#超参数
batch_size = 10
num_inputs = 28828
num_outputs = 10
num_hiddens = 256
learning_rate = .01

#加载数据
train_data ,test_data = load_data_fashion_mnist(batch_size)

#定义模型
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_hiddens,activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))
net.initialize()

#损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#训练
result= Result()
trainer = gluon.Trainer(net.collect_params(),'sgd',{"learning_rate":learning_rate})
num_batchs = len(train_data)
for epoch in range(10):
    train_loss = .0
    train_acc= .0
    for data,label in train_data:
        with ag.record() :
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
    print("Epoch %d : train_loss:%f, train_acc:%f, test_acc:%f"%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))

result.show()
