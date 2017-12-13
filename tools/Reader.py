import numpy as np
import tensorflow as tf

import Config as cfg
from scipy.io import wavfile
import wave
class Reader(object):
    def __init__(self):
        pass

    def read_data(self, file):
        self._rate, self._sig = wavfile.read(file)
    
    def get_rate(self):
        return self._rate
    
    def get_signal(self):
        return self._sig
   
class RecordReader(object):
    def __init__(self, filepath, level="frame", num_frames=300):
        super(RecordReader, self).__init__()
        assert level == "frame" or level == "frame", "yt8m-level must be 'frame' or 'video'"
        self.filepath = filepath
        self.level = level
        self.num_frames = num_frames
        self.reader = tf.TFRecordReader()
        