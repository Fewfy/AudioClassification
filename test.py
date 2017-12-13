import argparse
import os

import pafy
import tensorflow as tf

import init_path
import Config as config
from tools.Reader import Reader

if __name__ == '__main__':
    root = config.root
    wav_reader = Reader()
    for i in os.listdir(root):
        if os.path.splitext(i)[1] == '.wav':
            file = root + '/' + i
            wav_reader.read_data(file)
            print wav_reader.get_rate()
            print len(wav_reader.get_signal())
            break
            