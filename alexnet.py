"""Builds the OFFICE-31 network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

import data_loader

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('source_batch_size', 200,
                            """Number of source images to process in a batch.""")
tf.app.flags.DEFINE_integer('target_batch_size', 56,
                            """Number of target images to process in a batch.""")
tf.app.flags.DEFINE_string('source_tf_records_file', 'office31_data/office31/amazon.tfrecords',
                           """Path to the source data directory.""")
tf.app.flags.DEFINE_string('target_tf_records_file', 'office31_data/office31/webcam.tfrecords',
                           """Path to the target data directory.""")

# Global constants describing the OFFICE-31 data set.
IMAGE_SIZE = data_loader.IMAGE_SIZE
NUM_CLASSES = data_loader.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_loader.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_loader.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
LOW_LEARNING_RATE = 1e-3
HIGH_LEARNING_RATE = LOW_LEARNING_RATE * 10
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_EPOCHS_PER_DECAY = 10
LEARNING_RATE_DECAY_FACTOR = 0.1
train_layers = ['fc8']
MOVING_AVERAGE_DECAY = 0.9999  


class AlexNet(object):
  """Implementation of the AlexNet."""

  def __init__(self, x, keep_prob, num_classes, skip_layer, scope_initial):
    """Create the graph of the AlexNet model.
    Args:
      x: Placeholder for the input tensor.
      keep_prob: Dropout probability.
      num_classes: Number of classes in the dataset.
      skip_layer: List of names of the layer, that get trained from scratch.
      scope_initial: The initial of the scope name to separate the source
      and target variables from each other.
    """
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer
    self.scope_initial = scope_initial
    dirname = os.path.dirname(__file__)
    self.WEIGHTS_PATH = os.path.join(dirname, 'office31_data/bvlc_alexnet.npy')
    
    # Call the create function to build the computational graph of AlexNet
    self.create()

  def create(self):
    """Create the network graph."""
    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name=self.scope_initial + 'conv1')
    norm1 = lrn(conv1, 2, 1e-04, 0.75, name=self.scope_initial + 'norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name=self.scope_initial + 'pool1')
    
    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name=self.scope_initial + 'conv2')
    norm2 = lrn(conv2, 2, 1e-04, 0.75, name=self.scope_initial + 'norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name=self.scope_initial + 'pool2')
    
    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, 3, 3, 384, 1, 1, name=self.scope_initial + 'conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name=self.scope_initial + 'conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name=self.scope_initial + 'conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name=self.scope_initial + 'pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6*6*256])
    fc6 = fc(flattened, 6*6*256, 4096, name=self.scope_initial + 'fc6')
    dropout6 = dropout(fc6, self.KEEP_PROB)

    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, 4096, 4096, name=self.scope_initial + 'fc7')
    dropout7 = dropout(fc7, self.KEEP_PROB)

    # 8th Layer: FC and return unscaled activations
    self.fc8 = fc_8(dropout7, 4096, self.NUM_CLASSES, relu=False, name=self.scope_initial + 'fc8')


  def load_initial_weights(self, session):
    """Load weights from file into network.
    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    come as a dict of lists (e.g. weights['conv1'] is a list) and not as
    dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
    'biases') we need a special load function
    """
    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

      # Check if layer should be trained from scratch
      if op_name not in self.SKIP_LAYER:

        with tf.variable_scope(self.scope_initial + op_name, reuse=True):

          # Assign weights/biases to their corresponding tf variable
          for data in weights_dict[op_name]:

            # Biases
            if len(data.shape) == 1:
              var = tf.get_variable('biases', trainable=True)
              session.run(var.assign(data))

            # Weights
            else:
              var = tf.get_variable('weights', trainable=True)
              session.run(var.assign(data))


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
  """Create a convolution layer.
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k,
                                       strides=[1, stride_y, stride_x, 1],
                                       padding=padding)

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape=[filter_height,
                                                filter_width,
                                                input_channels/groups,
                                                num_filters])
    biases = tf.get_variable('biases', shape=[num_filters])

  if groups == 1:
    conv = convolve(x, weights)

  # In the cases of multiple groups, split inputs & weights and
  else:
    # Split input and weights and convolve them separately
    input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
    weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                             value=weights)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

    # Concat the convolved output together again
    conv = tf.concat(axis=3, values=output_groups)

  # Add biases
  bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

  # Apply relu function
  relu = tf.nn.relu(bias, name=scope.name)

  return relu


def fc(x, num_in, num_out, name, relu=True):
  """Create a fully connected layer."""
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out],
                              trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

  if relu:
    # Apply ReLu non linearity
    relu = tf.nn.relu(act)
    return relu
  else:
    return act


def fc_8(x, num_in, num_out, name, relu=True):
  """Create a fully connected layer."""
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out],
                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                              trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

  if relu:
    # Apply ReLu non linearity
    relu = tf.nn.relu(act)
    return relu
  else:
    return act
  

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
  """Create a max pooling layer."""
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides=[1, stride_y, stride_x, 1],
                        padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
  """Create a local response normalization layer."""
  return tf.nn.local_response_normalization(x, depth_radius=radius,
                                            alpha=alpha, beta=beta,
                                            bias=bias, name=name)


def dropout(x, keep_prob=0.5):
  """Create a dropout layer."""
  return tf.nn.dropout(x, keep_prob)


def inputs(source_data):
  """Construct input for Office31 training and evaluation using the Reader ops.
  Args:
    source_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.source_tf_records_file or not FLAGS.target_tf_records_file:
    raise ValueError('Please supply a tf_records_file')
  if source_data:
    images, labels = data_loader.inputs(filename=FLAGS.source_tf_records_file,
                                        batch_size=FLAGS.source_batch_size,
                                        num_examples_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
  else:
    images, labels = data_loader.inputs(filename=FLAGS.target_tf_records_file,
                                        batch_size=FLAGS.target_batch_size,
                                        num_examples_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
  return images, labels


def loss(source_scores, target_scores, source_labels, _lambda):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    source_scores, target_scores: Logits from inference().
    source_labels: Labels from distorted_inputs or inputs(). 2-D tensor
            of shape [batch_size]
    _lambda: A variable to trade off between coral and classification loss        
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  source_labels = tf.cast(source_labels, tf.int64)
  classification_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=source_labels, logits=source_scores))
  
  source_batch_size = tf.cast(tf.shape(source_scores)[0], tf.float32)
  target_batch_size = tf.cast(tf.shape(target_scores)[0], tf.float32)  
  d = tf.cast(tf.shape(source_scores)[1], tf.float32)
    
  # Source covariance
  xm = source_scores - tf.reduce_mean(source_scores, 0, keep_dims=True) 
  xc = tf.matmul(tf.transpose(xm), xm) / source_batch_size
  
  # Target covariance
  xmt = target_scores - tf.reduce_mean(target_scores, 0, keep_dims=True) 
  xct = tf.matmul(tf.transpose(xmt), xmt) / target_batch_size
    
  coral_loss = tf.reduce_sum(tf.multiply((xc - xct), (xc - xct)))
  coral_loss /= 4 * d * d 
  
  total_loss = classification_loss + _lambda * coral_loss
    
  tf.add_to_collection('losses', total_loss)

  # The total loss is defined as the classification loss plus the coral loss.
  return classification_loss, coral_loss, total_loss
   

def train(total_loss, global_step):
  """Train Office31 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """

  # Variables that affect learning rate.
  # num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / 200
  # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  # Decay the learning rate exponentially based on the number of steps.
  # low_lr = tf.train.exponential_decay(LOW_LEARNING_RATE,
  #                                 global_step,
  #                                 decay_steps,
  #                                 LEARNING_RATE_DECAY_FACTOR,
  #                                 staircase=True)
  # high_lr = tf.train.exponential_decay(HIGH_LEARNING_RATE,
  #                                 global_step,
  #                                 decay_steps,
  #                                 LEARNING_RATE_DECAY_FACTOR,
  #                                 staircase=True)
  # tf.summary.scalar('low_learning_rate', low_lr)

  # List of trainable variables of the layers we want to train
  low_lr_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] not in train_layers]
  high_lr_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers]
  
  # Compute gradients.
  with tf.control_dependencies(None):
    low_lr_var_opt = tf.train.MomentumOptimizer(LOW_LEARNING_RATE, MOMENTUM)
    high_lr_var_opt = tf.train.MomentumOptimizer(HIGH_LEARNING_RATE, MOMENTUM)
    low_lr_var_grads = low_lr_var_opt.compute_gradients(total_loss, low_lr_var_list)
    high_lr_var_grads = high_lr_var_opt.compute_gradients(total_loss, high_lr_var_list)

  # Apply gradients.
  apply_low_gradient_op = low_lr_var_opt.apply_gradients(low_lr_var_grads, global_step=global_step)
  apply_high_gradient_op = high_lr_var_opt.apply_gradients(high_lr_var_grads, global_step=global_step)  

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_low_gradient_op, apply_high_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op
