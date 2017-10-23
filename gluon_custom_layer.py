#coding：utf8
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from utils import load_data_fashion_mnist,accuracy,evaluate_accuracy,Result

batch_size = 10
num_inputs = 28*28
num_hiddens = 256
num_outputs = 10
learning_rate = .01

train_data,test_data = load_data_fashion_mnist(batch_size)

class MyLayer(gluon.nn.Block):
    def __init__(self,units,in_units,**kwargs):
        super(MyLayer,self).__init__(**kwargs)

        with self.name_scope():
            self.weight = self.params.get("weight",shape=(in_units,units))
            self.bias = self.params.get("bias",shape=(units,))


    def forward(self,x):
        X = nd.reshape(x,shape=(-1,self.weight.data().shape[0]))
        # print(X.shape)
        # print(self.weight.data().shape)
        # print(self.bias.data().shape)
        linar = nd.dot(X,self.weight.data())+self.bias.data()
        return nd.relu(linar)

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(MyLayer(num_hiddens,num_inputs))
    net.add(MyLayer(num_outputs,num_hiddens))

net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),'sgd',{"learning_rate":learning_rate})
for epoch in range(10):
    train_loss = .0
    train_acc = .0
    for data,label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net)
    print("Epoch %d train_loss:%f train_acc:%f test_acc:%f"%(epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))
