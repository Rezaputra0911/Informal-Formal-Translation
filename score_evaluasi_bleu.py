# -*- coding: utf-8 -*-
"""Score Evaluasi BLEU

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NVwSXFiHCZJLhg5b6zL9ZWefQZCvT4Pm
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

encoder_model100 = load_model('/content/drive/My Drive/Colab Notebooks/Skripsi/Encoder-Decoder/Embedding 100/encoder_model100.h5')
decoder_model100 = load_model('/content/drive/My Drive/Colab Notebooks/Skripsi/Encoder-Decoder/Embedding 100/decoder_model100.h5')

encoder_model200 = load_model('/content/drive/My Drive/Colab Notebooks/Skripsi/Encoder-Decoder/Embedding 200/encoder_model200.h5')
decoder_model200 = load_model('/content/drive/My Drive/Colab Notebooks/Skripsi/Encoder-Decoder/Embedding 200/decoder_model200.h5')

encoder_model300 = load_model('/content/drive/My Drive/Colab Notebooks/Skripsi/Encoder-Decoder/Embedding 300/encoder_model300.h5')
decoder_model300 = load_model('/content/drive/My Drive/Colab Notebooks/Skripsi/Encoder-Decoder/Embedding 300/decoder_model300.h5')

# Load idx2word_input
idx2word_input = np.load('/content/drive/My Drive/Colab Notebooks/Skripsi/Word2idx-Idx2word/idx2word_input.npy',allow_pickle='TRUE').item()
# Load idx2word_target
idx2word_target = np.load('/content/drive/My Drive/Colab Notebooks/Skripsi/Word2idx-Idx2word/idx2word_target.npy',allow_pickle='TRUE').item()
# Load word2idx_inputs
word2idx_inputs = np.load('/content/drive/My Drive/Colab Notebooks/Skripsi/Word2idx-Idx2word/word2idx_inputs.npy',allow_pickle='TRUE').item()
# Load word2idx_outputs
word2idx_outputs = np.load('/content/drive/My Drive/Colab Notebooks/Skripsi/Word2idx-Idx2word/word2idx_outputs.npy',allow_pickle='TRUE').item()

def translate_sentence(input_seq,enc,dec):
    states_value = enc.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<end>']
    output_sentence = []

    for _ in range(30):
        output_tokens, h, c = dec.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

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

  return encoder_input_sequences_test[1:2]

from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())

worksheet = gc.open('Dataset Kalimat').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

# Convert to a DataFrame and render.
import pandas as pd
colsname=['formal','non-formal']
df = pd.DataFrame.from_records(rows,columns=colsname)

"""#Data Training"""

df = df[:3000]
df.tail()

import re
def preprocess_aktual(w):
  w = w.lower()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = w.rstrip().strip()

  return w

def preprocess_input(w):
  w = w.lower()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", w)
  w = w.rstrip().strip()
  return w

for x in range(3):
  enc = 'encoder_model{}'.format((x+1)*100)
  dec = 'decoder_model{}'.format((x+1)*100)
  #aktual
  var_name1 = 'aktual{}'.format((x+1)*100)
  locals()[var_name1] = []
  for i in range(3000):
    locals()[var_name1].append(preprocess_aktual(df['formal'][i]))

  #masukkan
  var_name2 = 'masukkan{}'.format((x+1)*100)
  locals()[var_name2] = []
  for i in range(3000):
    locals()[var_name2].append(preprocess_input(df['non-formal'][i]))

  #respon
  var_name3 = 'respon{}'.format((x+1)*100)
  locals()[var_name3] = []
  for i in range(3000):
    input_teks = locals()[var_name2][i]
    input_seq = preproces(input_teks)
    terjemahan = translate_sentence(input_seq,locals()[enc],locals()[dec])
    locals()[var_name3].append(terjemahan)

for x in range(500):
  print("masukkan: ", masukkan100[x])
  print("respon :",respon100[x])
  print("aktual: ",aktual100[x])
  print('===============================')

for x in range(3):
  aktual = 'aktual{}'.format((x+1)*100)
  respon = 'respon{}'.format((x+1)*100)
  print('Embedding {}'.format((x+1)*100))
  score = 0
  for count in range(2000):
    score = score + sentence_bleu(locals()[aktual][count], locals()[respon][count])
  print ('Score : ', score/2000)
  print("=====================================================================================================")

for x in range(3):
  aktual = 'aktual{}'.format((x+1)*100)
  respon = 'respon{}'.format((x+1)*100)
  print('Embedding {}'.format((x+1)*100))
  print('BLEU-1: %f' % corpus_bleu(locals()[aktual][:2000], locals()[respon][:2000],weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(locals()[aktual][:2000], locals()[respon][:2000],weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(locals()[aktual][:2000], locals()[respon][:2000],weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(locals()[aktual][:2000], locals()[respon][:2000],weights=(0.25, 0.25, 0.25, 0.25)))
  print("=====================================================================================================")

"""#Data Testing"""

# Convert to a DataFrame and render.
import pandas as pd
colsname=['formal','non-formal']
df_test = pd.DataFrame.from_records(rows,columns=colsname)

df_test = df_test[3000:3100]
df_test = df_test.reset_index(drop=True)
df_test

for x in range(3):
  enc = 'encoder_model{}'.format((x+1)*100)
  dec = 'decoder_model{}'.format((x+1)*100)
  #aktual
  var_name1 = 'aktual_test{}'.format((x+1)*100)
  locals()[var_name1] = []
  for i in range(100):
    locals()[var_name1].append(preprocess_aktual(df_test['formal'][i]))

  #masukkan
  var_name2 = 'masukkan_test{}'.format((x+1)*100)
  locals()[var_name2] = []
  for i in range(100):
    locals()[var_name2].append(preprocess_input(df_test['non-formal'][i]))

  #respon
  var_name3 = 'respon_test{}'.format((x+1)*100)
  locals()[var_name3] = []
  for i in range(100):
    input_teks = locals()[var_name2][i]
    input_seq = preproces(input_teks)
    terjemahan = translate_sentence(input_seq,locals()[enc],locals()[dec])
    locals()[var_name3].append(terjemahan)

for x in range(100):
  print("masukkan: ", masukkan_test100[x])
  print("respon :",respon_test100[x])
  print("aktual: ",aktual_test100[x])
  print('===============================')

for x in range(3):
  aktual = 'aktual_test{}'.format((x+1)*100)
  respon = 'respon_test{}'.format((x+1)*100)
  print('Embedding {}'.format((x+1)*100))
  score = 0
  for count in range(100):
    score = score + sentence_bleu(locals()[aktual][count], locals()[respon][count])
  print ('Score : ', score/100)
  print("=====================================================================================================")

for x in range(3):
  aktual = 'aktual_test{}'.format((x+1)*100)
  respon = 'respon_test{}'.format((x+1)*100)
  print('Embedding {}'.format((x+1)*100))
  print('BLEU-1: %f' % corpus_bleu(locals()[aktual], locals()[respon],weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(locals()[aktual], locals()[respon],weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(locals()[aktual], locals()[respon],weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(locals()[aktual], locals()[respon],weights=(0.25, 0.25, 0.25, 0.25)))
  print("=====================================================================================================")

"""#Save File to TXT

##Training Data
"""

f=open('aktual_training.txt','w')
for x in range(len(aktual100)):
  s1='\n'.join(aktual100)
  f.write(s1)
f.close()

f=open('respon100_training.txt','w')
for x in range(len(respon100)):
  s1='\n'.join(respon100)
  f.write(s1)
f.close()

f=open('respon200_training.txt','w')
for x in range(len(respon200)):
  s1='\n'.join(respon200)
  f.write(s1)
f.close()

f=open('respon300_training.txt','w')
for x in range(len(respon300)):
  s1='\n'.join(respon300)
  f.write(s1)
f.close()

"""## Testing data"""

f=open('aktual_testing.txt','w')
for x in range(len(aktual_test100)):
  s1='\n'.join(aktual_test100)
  f.write(s1)
f.close()

f=open('respon_testing100.txt','w')
for x in range(len(respon_test100)):
  s1='\n'.join(respon_test100)
  f.write(s1)
f.close()

f=open('respon_testing200.txt','w')
for x in range(len(respon_test200)):
  s1='\n'.join(respon_test200)
  f.write(s1)
f.close()

f=open('respon_testing300.txt','w')
for x in range(len(respon_test300)):
  s1='\n'.join(respon_test300)
  f.write(s1)
f.close()