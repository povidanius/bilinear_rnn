import tensorflow as tf
import math
import numpy as np

def is_sparse(tensor):
    return isinstance(tensor, tf.SparseTensor)

def int_shape(x):    
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])


def ndim(x):    
    if is_sparse(x):
        return x._dims

    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def dot(x, y):   
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = (-1,) + int_shape(x)[1:]
        y_shape = int_shape(y)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if is_sparse(x):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out
