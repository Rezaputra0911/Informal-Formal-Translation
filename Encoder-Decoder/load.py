# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:44:53 2020

@author: Nathanael
"""

import numpy as np
from keras.models import load_model
import tensorflow as tf


def init(): 
    encoder_model = load_model('Embedding 200/encoder_model200.h5')
    decoder_model = load_model('Embedding 200/decoder_model200.h5')
    graph = tf.get_default_graph()
    return encoder_model,decoder_model,graph