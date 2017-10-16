#conding:utf8
from mxnet import gluon
from mxnet import ndarray as nd
def SGD(params,lr):
    for param in params:
        param[:] = param - lr*param.grad


def data_transform(data,label):
    return data.astype('float32')/255,label.astype('float32')

#加载数据
def data_transform(data,label):
    return data.astype('float32')/255,label.astype('float32')

def load_data_fashion_mnist(batch_size):
    mnist_train = gluon.data.vision.FashionMNIST(train=True,transform=data_transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False,transform=data_transform)

    ## 每次迭代产生batch_size个数据
    train_data = gluon.data.DataLoader(mnist_train,batch_size,shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test,batch_size,shuffle=False)
    return train_data,test_data


def accuracy(output,label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(data_iterator,net):
    total_acc =0.0
    for data,label in data_iterator:
        output = net(data)
        total_acc+=accuracy(output,label)
    return total_acc/len(data_iterator)
