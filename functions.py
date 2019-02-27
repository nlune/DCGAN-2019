import tensorflow as tf
import os
import shutil
import numpy as np

def extract_fn(tfrecord):
    # use with TFRecordDataset map function to get images from tfrecord
    # Extract features using the keys set during creation
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'rows': tf.FixedLenFeature([], tf.int64),
        'cols': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)

    image = tf.image.decode_image(sample['image'])
    img_shape = tf.stack([sample['rows'], sample['cols'], sample['channels']])
    filename = sample['filename']
    return image
