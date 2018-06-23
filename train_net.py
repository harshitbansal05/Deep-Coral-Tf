import tensorflow as tf
import os

from datetime import datetime
import time 

import convert_to_tfrecords
from alexnet import *

BATCH_SIZE = [200, 56]
EPOCHS = 40
NUM_CLASSES = 31
NUM_STEPS = 14 * EPOCHS 

train_layers = ['fc8']
keep_prob = 0.5

def train_():
  with tf.Graph().as_default():  
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for Office-31.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down
    with tf.device('/cpu:0'):
      source_images, source_labels = inputs(True)
      target_images, target_labels = inputs(False)
    
    _lambda = tf.placeholder(tf.float32)
    
    # Initialize model
    source_model = AlexNet(source_images, keep_prob, NUM_CLASSES, train_layers, 'source/')
    target_model = AlexNet(target_images, keep_prob, NUM_CLASSES, train_layers, 'target/')

    # Link variable to model output
    source_score = source_model.fc8
    target_score = target_model.fc8  
    
    # Op for calculating the loss
    classification_loss, coral_loss, loss_ = loss(source_score, target_score, source_labels, _lambda)
  
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = train(loss_, global_step)
        
    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname, 'office31_data/summary')
    writer = tf.summary.FileWriter(dirname)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:

      # Initialize all variables
      sess.run(tf.global_variables_initializer())

      # Add the model graph to TensorBoard
      writer.add_graph(sess.graph)

      # Load the pretrained weights into the non-trainable layer
      source_model.load_initial_weights(sess)
      target_model.load_initial_weights(sess)
    
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))
        _start_time = time.time()
        step = 0
        # Loop over number of epochs
        while step < NUM_STEPS and not coord.should_stop():
          # Initialize iterator with the training dataset
          cl_loss, cr_loss, loss_value, _ = sess.run([classification_loss, coral_loss, loss_, train_op], feed_dict={_lambda: (int(step / 14) + 1) / EPOCHS})
          step += 1
        
          if step % 10 == 0:
            current_time = time.time()
            duration = current_time - _start_time
            _start_time = current_time

            examples_per_sec = 10 * 200 / duration
            sec_per_batch = float(duration / 10)

            format_str = ('%s: step %d, classification_loss = %.2f, coral_loss = %.2f, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
            print (format_str % (datetime.now(), step, cl_loss, cr_loss, loss_value,
                               examples_per_sec, sec_per_batch))
          
          if step % 100 == 0:
            dirname = os.path.dirname(__file__)
            dest_directory = os.path.join(dirname, 'office31_data/train')
            checkpoint_name = os.path.join(dest_directory,
                                       'model_epoch'+str(step / 100)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)
            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))
            
          summary = tf.Summary()
          summary.ParseFromString(sess.run(merged_summary, feed_dict={_lambda: (int(step / 14) + 1) / EPOCHS}))
          writer.add_summary(summary, step)  
        
        dirname = os.path.dirname(__file__)
        dest_directory = os.path.join(dirname, 'office31_data/train')
        checkpoint_name = os.path.join(dest_directory,
                                       'model_epoch'+str(step / 100)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))
              
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)  


def main(_):
  convert_to_tfrecords.main()
  dirname = os.path.dirname(__file__)
  dest_directory = os.path.join(dirname, 'office31_data/train')
  if tf.gfile.Exists(dest_directory):
    tf.gfile.DeleteRecursively(dest_directory)
    tf.gfile.MakeDirs(dest_directory)
  train_()


if __name__ == '__main__':
  tf.app.run(main=main)
