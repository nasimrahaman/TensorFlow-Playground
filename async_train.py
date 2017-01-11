"""Async Training on MNIST"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.layers import Conv2D, Input
from keras.models import Model

from argparse import Namespace
import threading
import sys
import os

sys.path.append('/export/home/nrahaman/Python/Repositories/antipasti-tf/')

from Antipasti.io.runners import FeederRunner
from Antipasti.backend import image_tensor_to_matrix, reduce_

# ---- CONFIGURATION
print("[+] Configuration")
config = Namespace()

# Configure
config.verbose = True
config.num_epochs_per_data_thread = 100
config.num_data_threads = 2
config.batch_size = 100
config.devices = ('/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3')
# config.devices = ('/gpu:0',)
config.log_dir = '/export/home/nrahaman/Python/Scrap/tflogs/mnist-async'
config.checkpoint_dir = '/export/home/nrahaman/Python/Scrap/tfckpts/mnist-async'


# ---- DATA-LOGISTICS
print("[+] Data Logistics:")
# Load Data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# Extract numpy arrays from mnist to keep things explicit.
trX, trY = mnist.train.images.astype('float32'), mnist.train.labels.astype('float32')
teX, teY = mnist.test.images.astype('float32'), mnist.test.labels.astype('float32')
config.num_samples = trX.shape[0]

# Make generator and feeder runner
def data_generator():
    for epoch in range(config.num_epochs_per_data_thread):
        for idx in range(0, config.num_samples, config.batch_size):
            yield trX[idx:idx+config.batch_size].reshape(config.batch_size, 28, 28, 1), \
                  trY[idx:idx + config.batch_size].reshape(config.batch_size, 1, 1, 10)

print("[+] Building FeederRunner...")
feeder = FeederRunner(data_generator(), batch_size=50, dtypes=['float32', 'float32'],
                      num_threads=config.num_data_threads,
                      input_shape=[[None, 28, 28, 1], [None, 1, 1, 10]])
feeder.make_queue()
images, labels = feeder.dq()
print("[+] FeederRunners built.")

# ---- MODEL-DEFINITION
print("[+] Model Definition")

print("[+] Defining Model on CPU...")
# Define parameters on cpu
with tf.device('/cpu:0'):
    x = Input(shape=(28, 28, 1))
    y = Conv2D(10, 28, 28, border_mode='valid')(x)
    model = Model(input=x, output=y)


# Get outputs and train_ops on the gpu's
model_outputs = {}
model_losses = {}
train_ops = {}
model_accuracies = {}
summaries = {}

print("[+] Defining global step and learning_rate...")
global_step = tf.Variable(0., name='global_step')
learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step,
                                           decay_steps=500, decay_rate=0.7)

print("[+] Defining Model on devices...")
for device in config.devices:
    with tf.device(device):
        model_outputs[device] = model(images)
        logits_matrix = image_tensor_to_matrix(model_outputs[device])
        labels_matrix = image_tensor_to_matrix(labels)
        model_losses[device] = reduce_(tf.nn.softmax_cross_entropy_with_logits(logits=logits_matrix,
                                                                               labels=labels_matrix),
                                       mode='mean')
        model_accuracies[device] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_matrix, axis=1),
                                                                   tf.argmax(labels_matrix, axis=1)),
                                                          tf.float32))
        train_ops[device] = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9,
                                                       use_nesterov=True).\
            minimize(loss=model_losses[device], global_step=global_step)
        tf.add_to_collection('train_ops', train_ops[device])
    # Make summary
    loss_on_device_summary = tf.summary.scalar('loss_{}'.format(device.strip('/').replace(':', '')),
                                               model_losses[device])
    accuracy_on_device_summary = tf.summary.scalar('accuracy_{}'.
                                                   format(device.strip('/').replace(':', '')),
                                                   model_accuracies[device])
    summaries[device] = tf.summary.merge([loss_on_device_summary, accuracy_on_device_summary])
    tf.add_to_collection('summaries', summaries[device])

# ---- TRAINING
print("[+] Training:")
# Make session
session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
# Initialize all variables
session.run(tf.global_variables_initializer())

print("[+] Defining Summary Writer...")
# Get summary writer
summary_writer = tf.summary.FileWriter(logdir=config.log_dir)

print("[+] Defining Saver...")
# Define saver
saver = tf.train.Saver()

# Some calculations...
num_batches_per_data_thread = len(range(0, config.num_samples, config.batch_size)) * \
                              config.num_epochs_per_data_thread
num_batches = num_batches_per_data_thread * config.num_data_threads
num_batches_per_device = num_batches / len(config.devices)

# Make thread target
def step(sess, train_op, summary_op, num_steps):
    for step_num in range(num_steps):
        summary, _ = sess.run([summary_op, train_op])
        summary_writer.add_summary(summary=summary)

print("[+] Starting Runner...")
# Start runners
feeder.start_runner(session=session)

# Start training threads
training_threads = {}

print("[+] Starting Threads...")
for device in config.devices:
    training_threads[device] = threading.Thread(target=step,
                                                args=(session,
                                                      train_ops.get(device),
                                                      summaries.get(device),
                                                      num_batches_per_device))
    training_threads[device].start()

print("[+] Waiting for Threads to join...")
# Wait for threads to finish
for device in config.devices:
    training_threads[device].join()

print("[+] Saving...")
saver.save(session, save_path=os.path.join(config.checkpoint_dir, 'model'),
           global_step=global_step)

print("[+] Done.")