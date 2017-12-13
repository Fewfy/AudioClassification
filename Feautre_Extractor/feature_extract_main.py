# -*- utf-8 -*-
import csv
import os
import sys
import cv2
import feature_extractor
import numpy
import tensorflow as tf
from tensorflow import app
from tensorflow import flags

FLAGS = flags.FLAGS

CAP_PROP_POS_MSEC = 0

if __name__ == '__main__':
    flags.DEFINE_string('output_tfrecords_file', None, 'File containing tfrecords will be written at this path.')
    flags.DEFINE_string('input_videos_csv', None, 'video CSV file')
    flags.DEFINE_string('model_dir', os.path.join(os.getenv('HOME'), 'yt8m'), 'Directory to store model files.')
    flags.DEFINE_integer('frames_per_second', 1, 'This many frames per second will be processed')
    flags.DEFINE_string('labels_feature_key', 'labels', 'Labels will be written to context feature with this key')
    flags.DEFINE_string('image_feature_key', 'rgb', 'Image features will be written to sequence feature with this key')
    flags.DEFINE_string('video_file_key_feature_key', 'video_id', 'Input <video_file> will be written to context feature with this key')
    flags.DEFINE_boolean('insert_zero_audio_features', True, 'If set, inserts features with name "audio" to be 128-D')
    
def __int64_list_feature(int64_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _make_bytes(int_array):
    if bytes == str:
        return ''.join(map(chr, int_array))
    else:
        return bytes(int_array)
def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
    assert features.dtype == 'float32'
    assert len(features.shape) == 1
    features = numpy.clip(features, min_quantized_value, max_quantized_value)
    
    
def main(args):
    
    pass

if __name__ == '__main__':
    app.run(main)
