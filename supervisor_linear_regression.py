#coding:utf8
from mxnet import ndarray as nd
from mxnet import autograd as ag
import random
from utils import SGD

num_inputs = 4
num_examples = 10000
batch_size = 100
epochs = 100
learning_rate = 0.0001

true_w = [2,-3,4.2,-2.1]
true_b = 3.2
w = nd.random_normal(shape=(num_inputs,1))
b = nd.zeros((1,))
params = [w,b]


X = nd.random_normal(shape=(num_examples,num_inputs))
y = true_w[0]*X[:,0] + true_w[1]*X[:,1] + true_w[2]*X[:,2]+ true_w[3]*X[:,3] +true_b
y+=0.01*nd.random_normal(shape=y.shape)

def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0,num_examples,batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield nd.take(X,j) , nd.take(y,j)


for param in params:
    param.attach_grad()
    
#定义模型
def net(x):
    return nd.dot(x,w)+b
    
def square_loss(yhat,y):
    return (yhat-y.reshape(yhat.shape))**2
     
#def SGD(params,lr):
#    for param in params:
#        param[:] = param - lr*param.grad


for e in range(epochs):
    total_loss = 0 
    for data,label in data_iter():
        with ag.record():
            output = net(data)
            #print(output)
            #print(output.shape,label.shape)
            #break
            loss = square_loss(output,label)
        loss.backward()
        SGD(params,learning_rate)
        #print(params)
        
        total_loss += nd.sum(loss).asscalar()
    print("epoch %d ,average loss %f"%(e,total_loss/batch_size))
    
print("true_w ",true_w)  
print("true_b ",true_b)
print("learned_result ",params)

