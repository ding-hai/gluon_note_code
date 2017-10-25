#coding：utf8
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import init
from utils import load_data_fashion_mnist,accuracy,Result,SGD
import mxnet as mx


def evaluate_accuracy(data_iterator,net):
    total_acc =0.0
    for data,label in data_iterator:
        data = data.reshape((-1,1,resize,resize))
        output = net(data)
        total_acc+=accuracy(output,label)
    return total_acc/len(data_iterator)

batch_size = 128
resize = 224
learning_rate = 0.01
train_data,test_data = load_data_fashion_mnist(batch_size,resize)

nn = gluon.nn
net = nn.Sequential()
with net.name_scope():
    net.add(
    # 第一阶段
    nn.Conv2D(channels=96, kernel_size=11,
              strides=4, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 第二阶段
    nn.Conv2D(channels=256, kernel_size=5,
              padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 第三阶段
    nn.Conv2D(channels=384, kernel_size=3,
              padding=1, activation='relu'),
    nn.Conv2D(channels=384, kernel_size=3,
              padding=1, activation='relu'),
    nn.Conv2D(channels=256, kernel_size=3,
              padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 第四阶段
    nn.Flatten(),
    nn.Dense(4096, activation="relu"),
    nn.Dropout(.5),
    # 第五阶段
    nn.Dense(4096, activation="relu"),
    nn.Dropout(.5),
    # 第六阶段
    nn.Dense(10)
    )

net.initialize(init=init.Xavier())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': learning_rate})
num_batchs = len(train_data)
for epoch in range(100):
    train_loss = 0.0
    train_acc = 0.0
    for data,label in train_data:

        data = data.reshape(shape=(-1,1,resize,resize))
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        train_loss  += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net)
    print("Epoch:%d train_loss:%f train_acc:%f test_acc:%f"%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))
