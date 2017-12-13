import csv
import os
import sys
import tarfile
import numpy
import tensorflow as tf
from CNN.inception_v3 import Inception_V3

MODEL_DIR = os.path.join(os.getenv('HOME'), 'yt8m')
YT8M_PCA_MAT = 'http://data.yt8m.org/yt8m_pca.tgz'


class Youtube8MFeatureExtractor(object):
    def __init__(self, model_dir = MODEL_DIR):
        self.model_dir = model_dir
        self.inception_model = Inception_V3()
        
        self._inception_proto = os.path.join(self.model_dir, 'classify_image_graph_def.pb')
        
        self._load_inception(self._inception_proto)
        pass
    def extract_rgb_frame_features(self, frame_rgb, apply_pca=True):
        """Extract features from frames"""
        assert (len(frame_rgb) == 3)
        assert (frame_rgb.shape[2] == 3)
        
        with self._inception_graph.as_default():
            frame_features = self.session.run('pool_3/_reshape:0', feed_dict={'DecodeJpeg:0': frame_rgb})
            frame_features = frame_features[0]
        
        return frame_features
        
    def apply_pca(self, frame_features):
        pass
    def _maybe_download(self, url):
        pass
    def _load_inception(self, proto_file):
        graph_def = tf.GraphDef.FromString(open(proto_file, 'rb').read())
        self._inception_graph = tf.Graph()
        with self._inception_graph.as_default():
            _ = tf.import_graph_def(graph_def, name='')
            self.session = tf.Session()
        pass
        
    def _load_pca(self):
        pass
    
    