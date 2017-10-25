#coding:utf8
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from utils import load_data_fashion_mnist,accuracy,Result,SGD
import mxnet as mx
# import os
# os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"
#os.environ["MXNET_CPU_PRIORITY_NTHREADS"] = "8"


def evaluate_accuracy(data_iterator,net):
    total_acc =0.0
    for data,label in data_iterator:
        output = net(data,False)
        total_acc+=accuracy(output,label)
    return total_acc/len(data_iterator)


def pure_batch_norm(X,gamma,beta,is_training,moving_mean,moving_variance,eps=1e-5,moving_momen=0.9):
    length = len(X.shape)
    assert length in (2,4)
    if length == 2:
        mean = X.mean(axis=0)
        variance = ((X-mean)**2).mean(axis=0)
    else:
        mean = X.mean(axis=(0,2,3),keepdims=True)
        variance = ((X-mean)**2).mean(axis=(0,2,3),keepdims=True)
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(variance.shape)


    #归一化
    if is_training:
        moving_mean[:] = moving_momen*moving_mean+(1-moving_momen)
        moving_variance[:] =moving_momen*moving_variance+(1-moving_momen)*moving_variance
        X_hat =(X-mean)/nd.sqrt(variance+eps)
    else:
        X_hat = (X-moving_mean)/nd.sqrt(moving_variance+eps)
        print("X_hat: ",X_hat)

    return gamma.reshape(mean.shape)*X_hat+beta.reshape(mean.shape)

# 权重格式： out_channel, in_channel, conv_height, conv_weight
# 数据格式： batch_size, channel, img_height, img_weight
weight_scale = 0.01
c_1=20
W_1 = nd.random_normal(shape=(c_1,1,5,5),scale=weight_scale)
b_1 = nd.zeros(c_1)

gamma_1 = nd.random_normal(shape=c_1,scale=weight_scale)
beta_1 = nd.random_normal(shape=c_1,scale=weight_scale)
moving_mean_1 = nd.zeros(c_1)
moving_variance_1 = nd.zeros(c_1)

c_2 = 50
W_2 = nd.random_normal(shape=(c_2,c_1,3,3),scale=weight_scale)
b_2 = nd.zeros(c_2)

gamma_2 = nd.random_normal(shape=c_2,scale=weight_scale)
beta_2 = nd.random_normal(shape=c_2,scale=weight_scale)
moving_mean_2 = nd.zeros(c_2)
moving_variance_2 = nd.zeros(c_2)

o_3 = 128
W_3 = nd.random_normal(shape=(1250,o_3),scale=weight_scale)
b_3 = nd.zeros(o_3)

W_4 = nd.random_normal(shape=(o_3,10),scale=weight_scale)
b_4 = nd.zeros(10)

params = [W_1,b_1,gamma_1,beta_1,W_2,b_2,gamma_2,beta_2,W_3,b_3,W_4,b_4]

batch_size = 256
learning_rate = 0.2
train_data,test_data = load_data_fashion_mnist(batch_size)


#给参数分配存储梯度的内存
for param in params:
    param.attach_grad()

# 批处理层放在卷积层之后，激活层之前
def net(X,is_training=False):
    data = X.reshape(shape=(-1,1,28,28))
    h1_conv = nd.Convolution(data=data,weight=W_1,bias=b_1,kernel=W_1.shape[2:],num_filter=c_1)
    h1_batch_normal = pure_batch_norm(h1_conv,gamma_1,beta_1,is_training,moving_mean_1,moving_variance_1)
    h1_activation = nd.relu(h1_batch_normal)
    h1_out = nd.Pooling(data=h1_activation,pool_type="max",kernel=(2,2),stride=(2,2))

    h2_conv = nd.Convolution(data=h1_out,weight=W_2,bias=b_2,kernel=W_2.shape[2:],num_filter=c_2)
    h2_batch_normal = pure_batch_norm(h2_conv,gamma_2,beta_2,is_training,moving_mean_2,moving_variance_2)
    h2_activation = nd.relu(h2_batch_normal)
    h2_out = nd.Pooling(data=h2_activation,pool_type="max",kernel=(2,2),stride=(2,2))

    h2_flatten = nd.flatten(h2_out)

    #全连接
    h3_linar = nd.dot(h2_flatten,W_3)+b_3
    h3_out = nd.relu(h3_linar)

    h4_linar = nd.dot(h3_out,W_4)+b_4
    if not is_training:
        print("h4_linar ",h4_linar)
    return h4_linar

#损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# train
for epoch in range(30):
    train_loss = .0
    train_acc = .0
    num_batchs = len(train_data)
    for data,label in train_data:
        with ag.record():
            output = net(data,True)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        SGD(params,learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net)
    print("Epoch:%d train_loss:%f train_acc:%f test_acc:%f"%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))
