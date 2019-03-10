import tensorflow as tf
import os
import shutil
import numpy as np

def extract_fn(tfrecord):
    # use with TFRecordDataset map function to get images from tfrecord
    # Extract features using the keys set during creation
    features = {
        'image': tf.FixedLenFeature([], tf.string)
    }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)

    image = tf.image.decode_image(sample['image'])

    return image


def loadpb(filename, model_name='dcgan'):
    """Loads pretrained graph from ProtoBuf file
    Arguments:
        filename - path to ProtoBuf graph definition
        model_name - prefix to assign to loaded graph node names
    Returns:
        graph, graph_def - as per Tensorflow definitions
    """
    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            producer_op_list=None,
                            name=model_name)

    return graph, graph_def
