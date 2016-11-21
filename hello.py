"""Logistic Regression with TensorFlow and Gotchas for reluctant Theano Zealots"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from argparse import Namespace

config = Namespace()

# Configure
config.verbose = True
config.numepochs = 100
config.batchsize = 100

# Load Data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# Extract numpy arrays from mnist to keep things explicit.
trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels
config.numsamples = trX.shape[0]

# Initialize weight and bias variables.
W = tf.Variable(tf.random_normal(shape=[784, 10], mean=0., stddev=0.001), name='W')
b = tf.Variable(tf.zeros(shape=[10]), name='b')
# GOTCHA 1: These are equivalent to shared variables in theano.

# Initialize input and target label variables.
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
yt = tf.placeholder(tf.float32, shape=[None, 10], name='yt')
# GOTCHA 2: These would be theano.tensor.matrices in theano.

# Compute Logits.
yl = tf.matmul(x, W) + b
# GOTCHA 3: Unlike theano, broadcasting is numpy-esque and handled by tensorflow (i.e. no dimshuffle required).
# GOTCHA 4: tf.matmul replaces theano.tensor.dot (no shit sherlock)

# Compute output (for later evaluation)
y = tf.nn.softmax(yl)

# Get loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yl, labels=yt), name='L')
# GOTCHA 5: This is fairly similar to how one would do it theano, except for the novel-length variable names.

# Set up optimizer.
# Here's where the voodoo begins. In theano, we'd have to write the updates ourselves, which is nice. In tensorflow,
# the gradients are computed and the updates are applied, all in one fudging go.
momsgd = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True).minimize(loss)
# GOTCHA 6: To optimize w.r.t. only certain variables, put these variables in a list and pass it to minimize as the
#           keyword argument `var_list`. To do the same in theano, we'd do T.grad(loss, wrt=var_list) before
#           passing it to an update maker (e.g. lasagne.updates.blahblah).
# GOTCHA 7: To get symbolic gradient variables in tensorflow (similar to what you'd get from theano.tensor.grad), use
#           tf.gradients(loss, [wrt_var]).
# GOTCHA 8: sgd is the Op that applies the updates (i.e. what would usually go in as the updates keyword argument in a
#           theano function is now an op). If you need custom updates, it's probably possible to use tf.assign to assign
#           new values to tf.Variable's.

# This is an op to initialize all variables. No equivalent in Theano, I guess.
init = tf.initialize_all_variables()

# I guess session is the tensorflow 'instance' doing all the work.
with tf.Session() as sess:
    # Context managers, context managers everywhere

    # Run the op that initializes variables
    sess.run(init)
    # GOTCHA 9: with some imagination, sess.run is somewhat similar to the eval function in theano.
    #           If this were theano, we'd go init.eval(), but that's pushing it.

    # Iterate over epochs
    for epochnum in range(config.numepochs):
        # Init/Reset loss accumulator
        cumnumL = 0
        print("[.] Training epoch {} of {}:".format(epochnum, config.numepochs))
        # Loop over training samples
        for batchnum, idx in enumerate(range(0, config.numsamples, config.batchsize)):
            # Fetch batches
            xbatch = trX[idx:idx+config.batchsize]
            ybatch = trY[idx:idx+config.batchsize]

            # Train on a batch
            updateOp, numL = sess.run([momsgd, loss], feed_dict={x: xbatch, yt: ybatch})
            # GOTCHA 10: The feed_dict should be very reminiscent of theano's eval function. Here, we 'evaluate' both
            #            the update op (momsgd) and the loss simultaneously by passing them together in a list.

            # Update loss accumulator
            cumnumL += numL

        # Get average loss per sample
        meannumL = cumnumL/float(config.numsamples)

        if config.verbose:
            print("[i] Mean Training Loss (Epoch): {}".format(meannumL))

    # Done. Evaluate accuracy and call it a day.
    correct = tf.cast(tf.equal(tf.argmax(y, dimension=1), tf.argmax(yt, dimension=1)), dtype=tf.float32)
    accuracy = tf.reduce_mean(correct)

    print("[i] Final Test Accuracy: {}".format(accuracy.eval({x: teX, yt: teY})))
    # And here comes the cherry on top:
    # GOTCHA 10: Tensorflow has its own eval function.

    print("[+] Done.")


