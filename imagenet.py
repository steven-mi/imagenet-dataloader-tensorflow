###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  imagenet input pipeline
# Date:         11.2016
#
#
# TODO: 23 images are not jpeg and should be used with the according decoder.

""" Usage:
import tensorflow as tf
sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.imagenet import imagenet_data
  d = imagenet_data(batch_size=64, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
print(image_batch.shape)
print(target_batch.shape)
"""

import tensorflow as tf
import numpy as np
import threading

class imagenet_data:
  """
  Downloads the imagenet dataset and creates an input pipeline ready to be fed into a model.

  memory calculation:
    1 image is 299*299*3*4 bytes = ~1MB
    1024MB RAM = ~1000 images

  empirical memory usage with default config:
    TensorFlow +500MB
    imagenet_utils (loading all paths and labels) +400MB
    build input pipeline and fill queues +2.2GB

  - decodes jpg images
  - scales images into a uniform size
  - shuffles the input if specified
  - builds batches
  """
  NUM_THREADS = 8
  NUMBER_OF_CLASSES = 1000
  TRAIN_SET_SIZE = len(data.train_filenames) # 1281167 # ~250MB for string with paths
  TEST_SET_SIZE = len(data.val_filenames) # 50000
  IMAGE_HEIGHT = 299
  IMAGE_WIDTH = 299
  NUM_OF_CHANNELS = 3

  def __init__(self, batch_size, sess,
               filename_feed_size=200,
               filename_queue_capacity=800,
               batch_queue_capacity=1000,
               min_after_dequeue=1000,
               image_height=IMAGE_HEIGHT,
               image_width=IMAGE_WIDTH):
    """ Downloads the data if necessary. """
    print("Loading imagenet data")
    self.batch_size = batch_size
    self.filename_feed_size = filename_feed_size
    self.filename_queue_capacity = filename_queue_capacity
    self.batch_queue_capacity = batch_queue_capacity + 3 * batch_size
    self.min_after_dequeue = min_after_dequeue
    self.sess = sess
    self.IMAGE_HEIGHT = image_height
    self.IMAGE_WIDTH = image_width
    check_if_downloaded()

  def build_train_data_tensor(self, shuffle=False, augmentation=False):
    img_path, cls = load_training_data()
    return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation)

  def build_test_data_tensor(self, shuffle=False, augmentation=False):
    img_path, cls = load_test_data()
    return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation)

  def __build_generic_data_tensor(self, all_img_paths, all_targets, shuffle, augmentation):
    """
    Creates the input pipeline and performs some preprocessing.
    The full dataset needs to fit into memory for this version.
    """

    ## filename queue
    imagepath_input = tf.placeholder(tf.string, shape=[self.filename_feed_size])
    target_input = tf.placeholder(tf.float32, shape=[self.filename_feed_size])

    self.filename_queue = tf.FIFOQueue(self.filename_queue_capacity, [tf.string, tf.float32],
                                  shapes=[[], []])
    enqueue_op = self.filename_queue.enqueue_many([imagepath_input, target_input])
    single_path, single_target = self.filename_queue.dequeue()

    # one hot encode the target
    single_target = tf.cast(tf.sub(single_target, tf.constant(1.0)), tf.int32)
    single_target = tf.one_hot(single_target, depth=self.NUMBER_OF_CLASSES)

    # load the jpg image according to path
    file_content = tf.read_file(single_path)
    single_image = tf.image.decode_jpeg(file_content, channels=self.NUM_OF_CHANNELS)

    # convert to [0, 1]
    single_image = tf.image.convert_image_dtype(single_image,
                                                dtype=tf.float32,
                                                saturate=True)

    single_image = tf.image.resize_images(single_image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH])

    # Data Augmentation
    if augmentation:
      single_image = tf.image.resize_image_with_crop_or_pad(single_image, self.IMAGE_HEIGHT+4, self.IMAGE_WIDTH+4)
      single_image = tf.random_crop(single_image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_OF_CHANNELS])
      single_image = tf.image.random_flip_left_right(single_image)

      single_image = tf.image.per_image_standardization(single_image)

    if shuffle:
      images_batch, target_batch = tf.train.shuffle_batch([single_image, single_target],
                                                          batch_size=self.batch_size,
                                                          capacity=self.batch_queue_capacity,
                                                          min_after_dequeue=self.min_after_dequeue,
                                                          num_threads=self.NUM_THREADS)
    else:
      images_batch, target_batch = tf.train.batch([single_image, single_target],
                                                          batch_size=self.batch_size,
                                                          capacity=self.batch_queue_capacity,
                                                          num_threads=1)

    def enqueue(sess):
      under = 0
      max = len(all_img_paths)
      while not self.coord.should_stop():
        upper = under + self.filename_feed_size
        if upper <= max:
          curr_data = all_img_paths[under:upper]
          curr_target = all_targets[under:upper]
          under = upper
        else:
          rest = upper - max
          curr_data = np.concatenate((all_img_paths[under:max], all_img_paths[0:rest]))
          curr_target = np.concatenate((all_targets[under:max], all_targets[0:rest]))
          under = rest

        sess.run(enqueue_op, feed_dict={imagepath_input: curr_data,
                                        target_input: curr_target})

    enqueue_thread = threading.Thread(target=enqueue, args=[self.sess])

    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    enqueue_thread.isDaemon()
    enqueue_thread.start()

    return images_batch, target_batch

  def __del__(self):
    self.close()

  def close(self):
    self.filename_queue.close(cancel_pending_enqueues=True)
    self.coord.request_stop()
    self.coord.join(self.threads)


###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  Functions for loading the imagenet image paths and labels into memory.
# Date:         11.2016
#
#  In order to download the imagenet data you need to look at
#  utils/imagenet_download/run_me.sh
#

import tensorflow as tf
import random
import os

train_dir         = "data/train/"
validation_dir    = "data/validation/"
labels_file       = "data/imagenet_lsvrc_2015_synsets.txt"
metadata_file     = "data/imagenet_metadata.txt"
bounding_box_file = "data/imagenet_2012_bounding_boxes.csv"

###############################################################################
# Some TensorFlow Inception functions (ported to python3)
# source: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py

def _find_image_files(data_dir, labels_file):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
      Assumes that the ImageNet data set resides in JPEG files located in
      the following directory structure.
        data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
        data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
      where 'n01440764' is the unique synset label associated with these images.
    labels_file: string, path to the labels file.
      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        n01440764
        n01443537
        n01484850
      where each line corresponds to a label expressed as a synset. We map
      each synset contained in the file to an integer (based on the alphabetical
      ordering) starting with the integer 1 corresponding to the synset
      contained in the first line.
      The reason we start the integer labels at 1 is to reserve label 0 as an
      unused background class.
  Returns:
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  challenge_synsets = [l.strip() for l in
                       tf.gfile.FastGFile(labels_file, 'r').readlines()]

  labels = []
  filenames = []
  synsets = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for synset in challenge_synsets:
    jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    synsets.extend([synset] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(challenge_synsets)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  synsets = [synsets[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(challenge_synsets), data_dir))
  return filenames, synsets, labels

def _find_human_readable_labels(synsets, synset_to_human):
  """Build a list of human-readable labels.
  Args:
    synsets: list of strings; each string is a unique WordNet ID.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'
  Returns:
    List of human-readable strings corresponding to each synset.
  """
  humans = []
  for s in synsets:
    assert s in synset_to_human, ('Failed to find: %s' % s)
    humans.append(synset_to_human[s])
  return humans

def _find_image_bounding_boxes(filenames, image_to_bboxes):
  """Find the bounding boxes for a given image file.
  Args:
    filenames: list of strings; each string is a path to an image file.
    image_to_bboxes: dictionary mapping image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  Returns:
    List of bounding boxes for each image. Note that each entry in this
    list might contain from 0+ entries corresponding to the number of bounding
    box annotations for the image.
  """
  num_image_bbox = 0
  bboxes = []
  for f in filenames:
    basename = os.path.basename(f)
    if basename in image_to_bboxes:
      bboxes.append(image_to_bboxes[basename])
      num_image_bbox += 1
    else:
      bboxes.append([])
  print('Found %d images with bboxes out of %d images' % (
      num_image_bbox, len(filenames)))
  return bboxes

def _build_synset_lookup(imagenet_metadata_file):
  """Build lookup for synset to human-readable label.
  Args:
    imagenet_metadata_file: string, path to file containing mapping from
      synset to human-readable label.
      Assumes each line of the file looks like:
        n02119247    black fox
        n02119359    silver fox
        n02119477    red fox, Vulpes fulva
      where each line corresponds to a unique mapping. Note that each line is
      formatted as <synset>\t<human readable label>.
  Returns:
    Dictionary of synset to human labels, such as:
      'n02119022' --> 'red fox, Vulpes vulpes'
  """
  lines = tf.gfile.FastGFile(imagenet_metadata_file, 'r').readlines()
  synset_to_human = {}
  for l in lines:
    if l:
      parts = l.strip().split('\t')
      assert len(parts) == 2
      synset = parts[0]
      human = parts[1]
      synset_to_human[synset] = human
  return synset_to_human

def _build_bounding_box_lookup(bounding_box_file):
  """Build a lookup from image file to bounding boxes.
  Args:
    bounding_box_file: string, path to file with bounding boxes annotations.
      Assumes each line of the file looks like:
        n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940
      where each line corresponds to one bounding box annotation associated
      with an image. Each line can be parsed as:
        <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
      Note that there might exist mulitple bounding box annotations associated
      with an image file. This file is the output of process_bounding_boxes.py.
  Returns:
    Dictionary mapping image file names to a list of bounding boxes. This list
    contains 0+ bounding boxes.
  """
  lines = tf.gfile.FastGFile(bounding_box_file, 'r').readlines()
  images_to_bboxes = {}
  num_bbox = 0
  num_image = 0
  for l in lines:
    if l:
      parts = l.split(',')
      assert len(parts) == 5, ('Failed to parse: %s' % l)
      filename = parts[0]
      xmin = float(parts[1])
      ymin = float(parts[2])
      xmax = float(parts[3])
      ymax = float(parts[4])
      box = [xmin, ymin, xmax, ymax]

      if filename not in images_to_bboxes:
        images_to_bboxes[filename] = []
        num_image += 1
      images_to_bboxes[filename].append(box)
      num_bbox += 1

  print('Successfully read %d bounding boxes '
        'across %d images.' % (num_bbox, num_image))
  return images_to_bboxes

###############################################################################

class imagenet_data:
  synset_to_human = _build_synset_lookup(metadata_file)
  image_to_bboxes = _build_bounding_box_lookup(bounding_box_file)

  val_filenames, val_synsets, val_labels = _find_image_files(validation_dir, labels_file)
  train_filenames, train_synsets, train_labels = _find_image_files(train_dir, labels_file)
  humans = _find_human_readable_labels(val_synsets, synset_to_human)

def check_if_downloaded():
  if os.path.exists(train_dir):
    print("Train directory seems to exist")
  else:
    raise Exception("Train directory doesn't seem to exist.")

  if os.path.exists(validation_dir):
    print("Validation directory seems to exist")
  else:
    raise Exception("Validation directory doesn't seem to exist.")


def load_class_names():
  return data.humans

def load_training_data():
  return data.train_filenames, data.train_labels

def load_test_data():
  return data.val_filenames, data.val_labels

data = imagenet_data()



































