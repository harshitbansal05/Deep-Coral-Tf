import tensorflow as tf
import math
import os

from datetime import datetime

import data_loader
from alexnet import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 56,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('tf_records_file', 'office31_data/office31/webcam.tfrecords',
                           """Path to the data directory.""")

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_loader.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
NUM_CLASSES = 31
BATCH_SIZE = 56
train_layers = ['fc8']
keep_prob = 1


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    dirname = os.path.dirname(__file__)
    dest_directory = os.path.join(dirname, 'office31_data/train')
    ckpt = tf.train.get_checkpoint_state(dest_directory)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / BATCH_SIZE))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * BATCH_SIZE
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary)  
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(): 
  """Evaluate Office 31 for the entire webcam dataset"""
  with tf.Graph().as_default() as g:
    # Get images and labels for Office 31.
    images, labels = data_loader.inputs(FLAGS.tf_records_file,
                                      FLAGS.batch_size,
                                      NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                                      False)

    # Build a Graph that computes the logits predictions from the inference model.
    # Initialize model
    model = AlexNet(images, keep_prob, NUM_CLASSES, train_layers, 'source/')

    # Link variable to model output
    logits = model.fc8 
    labels = tf.reshape(labels, [BATCH_SIZE])

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    
    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    dirname = os.path.dirname(__file__)
    dest_directory = os.path.join(dirname, 'office31_data/eval')
    summary_writer = tf.summary.FileWriter(dest_directory, g)

    eval_once(saver, summary_writer, top_k_op, summary_op)


def main(_):
  dirname = os.path.dirname(__file__)
  dest_directory = os.path.join(dirname, 'office31_data/eval')
  if tf.gfile.Exists(dest_directory):
    tf.gfile.DeleteRecursively(dest_directory)
    tf.gfile.MakeDirs(dest_directory)
  evaluate()


if __name__ == '__main__':
  tf.app.run(main=main)