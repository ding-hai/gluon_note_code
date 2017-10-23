#coding：utf8
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

#卷积操作

## 权重的格式 out_channel, in_channel, height , width
w = nd.arange(8).reshape((1,2,2,2))
b = nd.array([1])

## 输入输出数据的格式 batch_size, channel, height, width
data = nd.arange(32).reshape((1,2,4,4))
convolution_output = nd.Convolution(data,w,b,kernel=w.shape[2:],num_filter=w.shape[0])
print(data)
print(convolution_output)
max_pooling_output = nd.Pooling(data=data,kernel=(2,2),pool_type='max',stride=(2,2))
avg_pooling_output = nd.Pooling(data=convolution_output,kernel=(2,2),pool_type='avg')
print("max_pooling_output=",max_pooling_output)
print("avg_pooling_output=",avg_pooling_output)
