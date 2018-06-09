import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import h5py
import random
from PIL import Image
import tensorflow as tf


class ExampleReader(object):
  def __init__(self, index, path_to_image_files):
    self.index = index
    self._path_to_image_files = path_to_image_files
    self._num_examples = len(self._path_to_image_files)
    self._example_pointer = 0

  @staticmethod
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  @staticmethod
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def read_and_convert(self):
    """Read and convert to example, returns None if no data is available.
    Args:
      nothing
    Returns:
      example: A serialized example prototype buffer  
    """
    if self._example_pointer == self._num_examples:
      return None
    path_to_image_file = self._path_to_image_files[self._example_pointer]
    self._example_pointer += 1

    image = Image.open(path_to_image_file)
    image = image.resize([227, 227])
    image = np.array(image).tobytes()
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': ExampleReader._bytes_feature(image),
        'label': ExampleReader._int64_feature(self.index)
    }))
    return example


def convert_to_tfrecords(path_to_dataset_dir, path_to_tfrecords_file):
  """Helper function to generate the tfrecords file from the path to 
  dataset directory. 
  Args:
    path_to_dataset_dir: path to the images directory
    path_to_tfrecords_file: path to create the tfrecords file
  Returns:
    num_examples: the number of examples in the tfrecords file  
  """  
  num_examples = 0
    
  writer = tf.python_io.TFRecordWriter(path_to_tfrecords_file)

  images_path = os.path.join(path_to_dataset_dir, 'images')
  type_images = sorted(os.listdir(images_path))
  for index, type_images_dir in enumerate(type_images):
    type_images_dir = os.path.join(images_path, type_images_dir)
    path_to_image_files = tf.gfile.Glob(os.path.join(type_images_dir, '*.jpg'))
    print('%d files found in %s' % (len(path_to_image_files), type_images_dir))

    example_reader = ExampleReader(index, path_to_image_files)
    for i, path_to_image_file in enumerate(path_to_image_files):
      print('(%d/%d) processing %s' % (i + 1, len(path_to_image_files), path_to_image_file))
      example = example_reader.read_and_convert()
      if example is None:
        break

      writer.write(example.SerializeToString())
      num_examples += 1

  writer.close()

  return num_examples


def convert_to_tf():
  """Helper function to generate the tfrecords file for source and
  target datasets.
  Args:
    nothing
  Returns:
    num_train_examples: the number of examples in the source dataset
    num_test_examples: the number of examples in the target dataset  
  """
  dirname = os.path.dirname(__file__)
  dest_directory = os.path.join(dirname, 'office31_data/office31')
  path_to_train_dir = os.path.join(dest_directory, 'amazon')
  path_to_test_dir = os.path.join(dest_directory, 'webcam')
  
  path_to_train_tfrecords_file = os.path.join(dest_directory, 'amazon.tfrecords')
  path_to_test_tfrecords_file = os.path.join(dest_directory, 'webcam.tfrecords')
  
  path_to_tfrecords_meta_file = os.path.join(dest_directory, 'meta.json')

  for path_to_file in [path_to_train_tfrecords_file, path_to_test_tfrecords_file]:
    assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

  print('Processing training data...')
  num_train_examples = convert_to_tfrecords(path_to_train_dir, path_to_train_tfrecords_file)
  print('Processing test data...')
  num_test_examples = convert_to_tfrecords(path_to_test_dir, path_to_test_tfrecords_file)
  print('Done')
  return num_train_examples, num_test_examples


def maybe_download_and_extract():
  """Download and extract the tarball from Office 31's website.
  Args:
    nothing
  Returns:
    nothing  
  """
  dirname = os.path.dirname(__file__)
  dest_directory = os.path.join(dirname, 'office31_data')
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory) 
  DATA_URL = 'https://github.com/SSARCandy/DeepCORAL/raw/master/dataset/office31.tar.gz'  
  filename = DATA_URL.split('/')[-1]
  extracted_filename = DATA_URL.split('/')[-1].split('.')[0]
  filepath = os.path.join(dest_directory, filename)
  extracted_filepath = os.path.join(dest_directory, extracted_filename)
  print(filepath)  
  if not os.path.exists(filepath) and not os.path.exists(extracted_filepath):
    print('Here')
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'office31')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main():
  maybe_download_and_extract()
  num_train_examples, num_test_examples = convert_to_tf()
