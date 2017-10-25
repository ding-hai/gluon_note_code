#coding：utf8
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from utils import load_data_fashion_mnist,accuracy,Result,SGD
import mxnet as mx

batch_size = 256
learning_rate = 0.2
train_data,test_data = load_data_fashion_mnist(batch_size)

def evaluate_accuracy(data_iterator,net):
    total_acc =0.0
    for data,label in data_iterator:
        data = data.reshape((-1,1,28,28))
        output = net(data)
        total_acc+=accuracy(output,label)
    return total_acc/len(data_iterator)

nn = gluon.nn
net = nn.Sequential()
with net.name_scope():
    net.add(
    nn.Conv2D(channels=20,kernel_size=5,activation="relu"),
    nn.MaxPool2D(pool_size=2,strides=2),
    nn.BatchNorm(axis=1),
    nn.Activation(activation='relu'),

    nn.Conv2D(channels=50,kernel_size=3,activation="relu"),
    nn.MaxPool2D(pool_size=2,strides=2),
    nn.BatchNorm(axis=1),
    nn.Activation(activation='relu'),

    nn.Flatten(),
    nn.Dense(128,activation="relu"),
    nn.Dropout(.2),
    nn.Dense(10)
    )
net.initialize()

#损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# train
trainer = gluon.Trainer(net.collect_params(),'sgd',{"learning_rate":learning_rate})
for epoch in range(200):
    train_loss = .0
    train_acc = .0
    num_batchs = len(train_data)
    for data,label in train_data:
        with ag.record():
            data = data.reshape(shape=(-1,1,28,28))
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net)
    print("Epoch:%d train_loss:%f train_acc:%f test_acc:%f"%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))
