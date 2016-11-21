"""Using Tensorflow on multiple devices (GPUs and CPUs)"""

import numpy as np
import tensorflow as tf
import time

# Task: compute A^n + B^n for:
n = 10
# Here's how we're gonna do this. GPU0 will compute A^n and B^n separately, and the CPU will sum the results to give
# A^n + B^n.

# Make bigass random matrices
A = np.random.uniform(size=(10000, 10000)).astype('float32')
B = np.random.uniform(size=(10000, 10000)).astype('float32')


# Define matrix^power
def matpow(M, n):
    # Abstract cases where n < 1
    if n < 1:
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))


# Context manager to tell Tensorflow where to put the graph. Constants `a` and `b` have a 'device' attribute that reads
# '/device:GPU:0', as does an and bn (see below).
with tf.device('/gpu:0'):
    # Wrap numpy arrays A and B
    a = tf.constant(A)
    b = tf.constant(B)
    # Compute:
    # a^n
    an = matpow(a, n)
    # ... and b ^ n
    bn = matpow(b, n)

# The summation A^n '+' B^n happens on the CPU, so:
with tf.device('/gpu:0'):
    sum_ = an + bn

tic = time.time()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Run the op.
    sess.run(sum_)
toc = time.time()

print("[+] Finished in {} seconds.".format(toc - tic))
# With CPU (20 core Xeon) add: [+] Finished in 109.710270882 seconds.
# FIXME: Why does this take that long?