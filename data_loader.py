"""Routine for decoding the Office 31 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

# Process images of this size. 
IMAGE_SIZE = 227
NUM_CLASSES = 31

# Global constants describing the Office31 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2817
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 795


def read_office31(filename_queue):
  """Reads and parses examples from Office31 data files.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (IMAGE_SIZE)
      width: number of columns in the result (IMAGE_SIZE)
      depth: number of color channels in the result (3)
      label: an int32 Tensor with the label in the range 0..30
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class Office31Record(object):
    pass
  result = Office31Record()

  # Dimensions of the images in the Office31 dataset.
  result.height = IMAGE_SIZE
  result.width = IMAGE_SIZE
  result.depth = 3
  
  # Read a record, getting filenames from the filename_queue.  
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64)
    })
  image = tf.decode_raw(features['image'], tf.uint8)
  result.uint8image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  result.label = features['label']
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type tf.float32.
    label: 1-D Tensor of type tf.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, labels


def inputs(filename, batch_size, num_examples_per_epoch, shuffle=True):
  """Construct input for Office31 training using the Reader ops.
  Args:
    filename: Path to the Office31 TF Records data directory.
    batch_size: Number of images per batch.
    num_examples_per_epoch: The number of examples per epoch
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  dirname = os.path.dirname(__file__)
  print(dirname)
  file_name = os.path.join(dirname, filename)
  if not tf.gfile.Exists(file_name):
    raise ValueError('Failed to find file: ' + file_name)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer([file_name])

  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_office31(filename_queue)
    float_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_stddev = tf.constant([0.229, 0.224, 0.225])
    
    minus_mean1 = tf.fill([IMAGE_SIZE, IMAGE_SIZE, 1], -imagenet_mean[0])
    minus_mean2 = tf.fill([IMAGE_SIZE, IMAGE_SIZE, 1], -imagenet_mean[1])
    minus_mean3 = tf.fill([IMAGE_SIZE, IMAGE_SIZE, 1], -imagenet_mean[2])
    
    stddev1 = tf.fill([IMAGE_SIZE, IMAGE_SIZE, 1], imagenet_stddev[0])
    stddev2 = tf.fill([IMAGE_SIZE, IMAGE_SIZE, 1], imagenet_stddev[1])
    stddev3 = tf.fill([IMAGE_SIZE, IMAGE_SIZE, 1], imagenet_stddev[2])
    
    minus_mean = tf.concat([minus_mean1, minus_mean2, minus_mean3], axis=2)
    stddev = tf.concat([stddev1, stddev2, stddev3], axis=2)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image.set_shape([height, width, 3])
    float_image = float_image / 255.0
    float_image = float_image + minus_mean
    float_image = float_image / stddev
    
    label = tf.reshape(read_input.label, [1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 1
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    
    print ('Filling queue with %d Office 31 images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label, min_queue_examples,
                                         batch_size, shuffle)
