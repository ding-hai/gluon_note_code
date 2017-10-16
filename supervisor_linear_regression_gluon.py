#coding:utf8
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon


num_inputs = 4
num_examples = 1000
batch_size = 10
epochs = 100
learning_rate = 0.001

true_w = [2,-3,4.2,-2.1]
true_b = 3.2
w = nd.random_normal(shape=(num_inputs,1))
b = nd.zeros((1,))
params = [w,b]

X = nd.random_normal(shape=(num_examples,num_inputs))
y = true_w[0]*X[:,0] + true_w[1]*X[:,1] + true_w[2]*X[:,2]+ true_w[3]*X[:,3] +true_b
y+=0.01*nd.random_normal(shape=y.shape)

dataset = gluon.data.ArrayDataset(X,y)
data_iter = gluon.data.DataLoader(dataset,batch_size,shuffle=True)

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize()
square_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':learning_rate})
for e in range(epochs):
    total_loss = 0
    for data,label in data_iter:
        with ag.record():
            output = net(data)
            loss = square_loss(output,label)
        loss.backward()
        trainer.step(batch_size)
        total_loss+=nd.sum(loss).asscalar()
    print("epoch:%d average loss:%f"%(e,total_loss/batch_size))
dense = net[0]
print("true w",true_w)
print("learned result weight:",dense.weight.data())

print("true b",true_b)
print("learned result bias:",dense.bias.data())


