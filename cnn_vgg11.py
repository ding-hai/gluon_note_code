#coding：utf8
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import init
from utils import load_data_fashion_mnist,accuracy,Result,SGD
import mxnet as mx

batch_size = 128
resize = 96
num_outputs = 10
learning_rate = 0.01
train_data,test_data = load_data_fashion_mnist(batch_size,resize)


def evaluate_accuracy(data_iterator,net):
    total_acc =0.0
    for data,label in data_iterator:
        data = data.reshape((-1,1,resize,resize))
        output = net(data)
        total_acc+=accuracy(output,label)
    return total_acc/len(data_iterator)


def vgg_block(conv_nums,channels):
    net = gluon.nn.Sequential()
    for _ in range(conv_nums):
        net.add(gluon.nn.Conv2D(channels=channels,kernel_size=3,padding=1,activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2,strides=2))
    return net

def vgg_stack(architecture):
    net = gluon.nn.Sequential()
    for (conv_nums,channels) in architecture:
        net.add(vgg_block(conv_nums,channels))
    return net

architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(
    vgg_stack(architecture),
    gluon.nn.Flatten(),
    gluon.nn.Dense(4096,activation="relu"),
    gluon.nn.Dropout(0.4),
    gluon.nn.Dense(4096,activation="relu"),
    gluon.nn.Dropout(0.4),
    gluon.nn.Dense(num_outputs)
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
    print("VGG11: Epoch:%d train_loss:%f train_acc:%f test_acc:%f"%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))
