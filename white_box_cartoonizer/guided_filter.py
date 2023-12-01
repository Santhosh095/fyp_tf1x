import tensorflow as tf
import numpy as np


def tf_box_filter(x, r):
    k_size = int(2*r+1)
    ch = x.get_shape().as_list()[-1]
    weight = 1/(k_size**2)
    box_kernel = weight*np.ones((k_size, k_size, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.compat.v1.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    
    x_shape = tf.compat.v1.shape(x)

    N = tf_box_filter(tf.compat.v1.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = tf.compat.v1.add(mean_A * x, mean_b, name='final_add')

    return output


def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
       
    lr_x_shape = tf.compat.v1.shape(lr_x)

    hr_x_shape = tf.compat.v1.shape(hr_x)
    
    N = tf_box_filter(tf.compat.v1.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

    mean_x = tf_box_filter(lr_x, r) / N
    mean_y = tf_box_filter(lr_y, r) / N
    cov_xy = tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf.compat.v1.image.resize_images(A, hr_x_shape[1: 3])
    mean_b = tf.compat.v1.image.resize_images(b, hr_x_shape[1: 3])

    output = mean_A * hr_x + mean_b
    
    return output


if __name__ == '__main__':
    pass