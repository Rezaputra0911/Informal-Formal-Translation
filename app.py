# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:29:05 2020

@author: Nathanael
"""
from flask import Flask, render_template, request, jsonify
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger('root')


global graph, session
session = tf.Session()
graph = tf.get_default_graph()
with graph.as_default():
    with session.as_default():
        logging.info("neural network initialised")


#init flask app
app = Flask(__name__)


def get_model(): 
    global encoder_model, decoder_model, idx2word_input, idx2word_target, word2idx_inputs, word2idx_outputs
    with graph.as_default():
        with session.as_default():
            
            encoder_model = load_model("Encoder-Decoder/Embedding 200/encoder_model200.h5")
            encoder_model.load_weights("Encoder-Decoder/Embedding 200/encoder_model200.h5")
            decoder_model = load_model("Encoder-Decoder/Embedding 200/decoder_model200.h5")
            decoder_model.load_weights("Encoder-Decoder/Embedding 200/decoder_model200.h5")
            
            # Load idx2word_input
            idx2word_input = np.load('Word2idx-Idx2word/idx2word_input.npy',allow_pickle='TRUE').item()
            # Load idx2word_target
            idx2word_target = np.load('Word2idx-Idx2word/idx2word_target.npy',allow_pickle='TRUE').item()
            # Load word2idx_inputs
            word2idx_inputs = np.load('Word2idx-Idx2word/word2idx_inputs.npy',allow_pickle='TRUE').item()
            # Load word2idx_outputs
            word2idx_outputs = np.load('Word2idx-Idx2word/word2idx_outputs.npy',allow_pickle='TRUE').item()

get_model()

def translate_sentence(input_seq):
    with graph.as_default():
        with session.as_default():
            states_value = encoder_model.predict(input_seq)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = word2idx_outputs['<sos>']
            eos = word2idx_outputs['<end>']
            output_sentence = []
            for _ in range(28):
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
                idx = np.argmax(output_tokens[0, 0, :])
                if eos == idx:
                    break
                word = ''
                if idx > 0:
                    word = idx2word_target[idx]
                    output_sentence.append(word)
                target_seq[0, 0] = idx
                states_value = [h, c]
            print("debug 3")
            print('Translate() Ouput_sentences (Python)')
            return ' '.join(output_sentence)  

def preproces(input):
  from nltk.tokenize import RegexpTokenizer
  tokenizer = RegexpTokenizer(r'\w+')
  kata_input = tokenizer.tokenize(input)
  #tokenisasi
  test_word2idx = []
  for x in kata_input:
    try:
      test_word2idx.append(word2idx_inputs[x])
    except:
      test_word2idx.append(word2idx_inputs['unk']) 
  # membuat list bantu
  list_bantu = [x for x in range(28)]
  # array untuk encoder
  jadi = []
  jadi.append(list_bantu)
  jadi.append(test_word2idx)
  encoder_input_sequences_test = pad_sequences(jadi, maxlen=28)
  print("debug 2")
  print('Preproces() Encoder_input_sequences (Python)')
  return encoder_input_sequences_test[1:2]

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predict')       
def predict():
    kalimat = request.args.get('b')
    print("debug 1")
    print('Pengiriman Data Kalimat request.script_root() (HTML)')
    print('Penerimaan Data Kalimat request.args.get() (Python)')
    input_seq = preproces(kalimat)
    terjemahan = translate_sentence(input_seq)
    print('debug 4')
    print('Pengiriman kalimat hasil prediksi Jsonify() (Python)')
    print('Penerimaan kalimat hasil prediksi Function(data).text() (HTML)')
    return jsonify(result=terjemahan)
    

if __name__ == '__main__':
    app.run(debug=True)
    # kalimat = "enggak bakal bisa pake cara itu"
    # input_seq = preproces(kalimat)
    # terjemahan = translate_sentence(input_seq)
    # print(terjemahan)
    
