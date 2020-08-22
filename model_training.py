# -*- coding: utf-8 -*-
"""Model Training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1POfivmQKY-RURrZUdei9JLV9Wv9dTmRt
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

"""Sebagai langkah pertama, kami akan mengimpor pustaka yang diperlukan dan akan mengonfigurasi nilai untuk berbagai parameter yang akan kami gunakan dalam kode. Pertama-tama mari kita mengimpor perpustakaan yang diperlukan:"""

import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 200
LSTM_NODES =256
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 300

"""#Dataset

Model terjemahan bahasa yang akan kami kembangkan dalam artikel ini akan menerjemahkan kalimat non-formal ke dalam kalimat formal dari bahasa Indonesia. 

Untuk mengembangkan model seperti itu, kita membutuhkan dataset yang berisi kalimat non-formal dan terjemahannya dalam kalimat formal.
"""

from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())

worksheet = gc.open('Dataset Kalimat').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

"""Dataset tersebut berada pada google sheet dengan nama file "Dataset Kalimat". Dengan menggunakan Dataframe dari Pandas, data tersebut dipisah menjadi 2 kolom dengan nama kolom "formal" dan "non-formal""""

# Convert to a DataFrame and render.
import pandas as pd
colsname=['formal','non-formal']
df = pd.DataFrame.from_records(rows,columns=colsname)

df = df[:3000]
df

# Convert to a DataFrame and render.
import pandas as pd
colsname=['formal','non-formal']
df_emb = pd.DataFrame.from_records(rows,columns=colsname)
df_emb = df_emb[:3100]
df_emb

"""# Data Preprocessing

Model terjemahan mesin saraf sering didasarkan pada arsitektur seq2seq. Arsitektur seq2seq adalah arsitektur encoder-decoder yang terdiri dari dua jaringan LSTM: encoder LSTM dan decoder LSTM. Input ke encoder LSTM adalah kalimat dalam bahasa aslinya; input ke decoder LSTM adalah kalimat dalam bahasa yang diterjemahkan dengan token awal kalimat. Outputnya adalah kalimat target aktual dengan token akhir kalimat.

Dalam dataset kami, kami tidak perlu memproses input, namun, kami perlu membuat dua salinan dari kalimat yang diterjemahkan: satu dengan token awal kalimat dan yang lainnya dengan token akhir kalimat. Berikut ini skrip yang melakukan itu:
"""

import re

def input_sen(w):
  w = w.lower()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", w)
  w = w.rstrip().strip()

  return w

def output_sen(w):
  w = w.lower()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", w)
  w = w.rstrip().strip()
  w = w + ' <end>'
  return w

def output_sen_in(w):
  w = w.lower()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", w)
  w = w.rstrip().strip()
  w = '<sos> ' + w
  return w

input_sentences = []
for x in range(3000):
  input_sentences.append(input_sen(df['non-formal'][x]))

for x in range(500):
  print(input_sentences[x])

output_sentences = []
for x in range(3000):
  output_sentences.append(output_sen(df['formal'][x]))

for x in range(500):
  print(output_sentences[x])

output_sentences_inputs = []
for x in range(3000):
  output_sentences_inputs.append(output_sen_in(df['formal'][x]))

for x in range(500):
  print(output_sentences_inputs[x])

"""Pada skrip di atas kita membuat tiga daftar input_sentences [], output_sentences [], dan output_sentences_inputs []. 

Selanjutnya, dalam for loop tiap baris pada dataframe dibaca baris demi baris. 

input_sentences [] merupakan kalimat dari kolom non-formal dengan pengolahan kata.
output_sentences [] merupakan kalimat dari kolom formal dengan pengolahan kata ditambah token <end>.
output_sentences_inputs [] merupakan kalimat kolom formal dengan pengolahan kata ditambah token <sos>.

Token <end>, yang menandai akhir kalimat diawali dengan kalimat yang diterjemahkan, dan kalimat yang dihasilkan ditambahkan ke daftar output_sentences []. 
Demikian pula, token <sos>, yang merupakan singkatan dari "start of kalimat", digabungkan pada awal kalimat yang diterjemahkan dan hasilnya ditambahkan ke daftar output_sentences_inputs []. 

Simpul berakhir jika jumlah kalimat yang ditambahkan ke daftar lebih besar dari variabel NUM_SENTENCES.

Akhirnya jumlah sampel dalam tiga daftar ditampilkan dalam output:
"""

print("num samples input:", len(input_sentences))
print("num samples output:", len(output_sentences))
print("num samples output input:", len(output_sentences_inputs))

"""print("num samples input:", len(input_sentences))
print("num samples output:", len(output_sentences))
print("num samples output input:", len(output_sentences_inputs))
"""

for i in range(200,250):
  print(input_sentences[i])
  print(output_sentences[i])
  print(output_sentences_inputs[i])
  print("-----------------------------------------")

"""# Tokenization and Padding

Langkah selanjutnya adalah tokenizing kalimat asli dan terjemahan dan menerapkan padding pada kalimat yang lebih panjang atau lebih pendek dari panjang tertentu, yang dalam hal input akan menjadi panjang dari kalimat input terpanjang. Dan untuk output, ini akan menjadi panjang kalimat terpanjang dalam output.

Untuk tokenization, kelas Tokenizer dari pustaka keras.preprocessing.text dapat digunakan. Kelas tokenizer melakukan dua tugas:

     Ini membagi kalimat menjadi daftar kata yang sesuai
     Kemudian mengubah kata menjadi bilangan bulat

Ini sangat penting karena pembelajaran yang dalam dan algoritma pembelajaran mesin bekerja dengan angka. Skrip berikut digunakan untuk menandai token input kalimat:
"""

input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

word2idx_inputs = input_tokenizer.word_index
print('Total unique words in the input: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
print("Length of longest sentence in input: %g" % max_input_len)
print(input_integer_seq[100])

print(input_sentences[100])
print(input_integer_seq[100])

"""Selain tokenization dan konversi integer, atribut word_index dari kelas Tokenizer mengembalikan kamus kata-ke-indeks di mana kata-kata adalah kunci dan bilangan bulat yang sesuai adalah nilainya. Script di atas juga mencetak jumlah kata unik dalam kamus dan panjang kalimat terpanjang dalam input:

Demikian pula, kalimat-kalimat output juga dapat tokenized dengan cara yang sama seperti yang ditunjukkan di bawah ini:
"""

output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)
print(output_integer_seq)
word2idx_outputs = output_tokenizer.word_index
print('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
print('num_word_outputs:',num_words_output)
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: %g" % max_out_len)

"""Dari perbandingan jumlah kata unik dalam input dan output, dapat disimpulkan bahwa kalimat bahasa Inggris biasanya lebih pendek dan mengandung lebih sedikit jumlah kata rata-rata, dibandingkan dengan kalimat bahasa Prancis yang diterjemahkan.

Selanjutnya, kita perlu mengisi input. Alasan di balik pengisian input dan output adalah bahwa kalimat teks dapat memiliki panjang yang berbeda-beda, namun LSTM (algoritma yang akan kita latih model kita) mengharapkan instance input dengan panjang yang sama. Karena itu, kita perlu mengubah kalimat kita menjadi vektor dengan panjang tetap. Salah satu cara untuk melakukan ini adalah melalui padding.

Dalam padding, panjang tertentu didefinisikan untuk kalimat. Dalam kasus kami, panjang kalimat terpanjang dalam input dan output akan digunakan untuk masing-masing kalimat input dan output. Kalimat terpanjang dalam input berisi 6 kata. Untuk kalimat yang mengandung kurang dari 6 kata, nol akan ditambahkan dalam indeks kosong. Script berikut berlaku padding ke kalimat input.
"""

encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
print("encoder_input_sequences.shape:", encoder_input_sequences.shape)
print("encoder_input_sequences[6]:", encoder_input_sequences[6])

colsname = ["kalimat","input integer","encoder sequence"]
en_in = pd.DataFrame.from_records(list(zip(input_sentences,input_integer_seq,encoder_input_sequences)),columns=colsname)

en_in.tail()

"""Karena ada 1.100 kalimat dalam input dan setiap kalimat input panjang 25, bentuk input sekarang (1100, 25). Jika Anda melihat urutan bilangan bulat untuk kalimat pada indeks 6 dari kalimat input, Anda dapat melihat bahwa ada 8 nol, diikuti oleh nilai 1107,1109,5 dan seterusnya. Anda mungkin ingat bahwa kalimat asli pada indeks 6 kata pertama dan kedua "rino dan "menjelaskan". 
Tokenizer membagi kalimat menjadi beberapa kata, mengubahnya menjadi bilangan bulat, dan kemudian menerapkan pra-padding dengan menambahkan tiga nol pada awal urutan bilangan bulat yang sesuai untuk kalimat pada indeks 6 dari daftar input.

Untuk memverifikasi bahwa nilai integer untuk "rino" dan "menjelaskan" masing-masing adalah 1107 dan 1109, Anda bisa meneruskan kata-kata ke kamus word2index_inputs, seperti yang ditunjukkan di bawah ini:
"""

print(word2idx_inputs["yang"])
print(word2idx_inputs["menjelaskan"])

"""Save word2idx_inputs"""

# Save
np.save('word2idx_inputs.npy', word2idx_inputs)

decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
print("decoder_input_sequences.shape:", decoder_input_sequences.shape)
print("decoder_input_sequences[6]:", decoder_input_sequences[6])

"""Kalimat pada 4 kata pertama di indeks 6 dari input dekoder adalah "<sos> rino menjelaskan, ... " 

Jika Anda mencetak bilangan bulat yang sesuai dari kamus word2idx_outputs, Anda akan melihat 4, 1126, 517, dan 1 dicetak pada konsol, seperti yang ditunjukkan di sini:
"""

colsname = ["kalimat","output integer","decoder sequence"]
de_in = pd.DataFrame.from_records(list(zip(output_sentences,output_integer_seq,decoder_input_sequences)),columns=colsname)

de_in.tail()

print(word2idx_outputs["<sos>"])
print(word2idx_outputs["rino"])
print(word2idx_outputs["menjelaskan"])
print(word2idx_outputs[","])

"""Save word2idx_outputs"""

# Save
np.save('word2idx_outputs.npy', word2idx_outputs)

"""Lebih lanjut penting untuk menyebutkan bahwa dalam kasus dekoder, post-padding diterapkan, yang berarti bahwa nol ditambahkan pada akhir kalimat. 

Di encoder, angka nol telah diisi di awal. Alasan di balik pendekatan ini adalah bahwa output encoder didasarkan pada kata-kata yang terjadi pada akhir kalimat, oleh karena itu kata-kata asli disimpan di akhir kalimat dan nol padded di awal. 

Di sisi lain, dalam kasus decoder, pemrosesan dimulai dari awal kalimat, dan oleh karena itu post padding dilakukan pada input dan output decoder.

# Word2Vec


Karena kita menggunakan model deep learning, dan model depp learning bekerja dengan angka, maka kita perlu mengubah kata-kata kita menjadi representasi vektor numerik yang sesuai. Tapi kami sudah mengubah kata-kata kami menjadi bilangan bulat. Jadi apa perbedaan antara representasi integer dan embedding kata?

Ada dua perbedaan utama antara representasi integer tunggal dan embedding kata. Dengan reprensentasi integer, sebuah kata diwakili hanya dengan integer tunggal. Dengan representasi vektor, sebuah kata diwakili oleh vektor 50, 100, 200, atau dimensi apa pun yang Anda suka. Karenanya, embeddings kata menangkap lebih banyak informasi tentang kata-kata. Kedua, representasi bilangan bulat tunggal tidak menangkap hubungan antara kata-kata yang berbeda. Sebaliknya, kata embeddings mempertahankan hubungan antara kata-kata. Anda dapat menggunakan embeddings kata kustom atau Anda dapat menggunakan embeddings kata pretrained.

pada penelitian ini, embedding dilakukan secara manual dari kalimat dalam dataset.  Dengan bantuan gensim.models, kita membangun word2vec secara mudah dan cepat.
"""

import nltk
nltk.download('punkt')

#tokenize, dengan menghilangkan punctuation, dan mengubah digit menjadi #

from nltk import word_tokenize, sent_tokenize
import re
import string

puncts = string.punctuation
print(puncts)
    
sentences = []
for y in range(df_emb.shape[0]):
    doc = sent_tokenize(df_emb['formal'][y])
    sentence=[sent for sent in doc]
    for x in range(len(sentence)):
        words = [t for t in word_tokenize(sentence[x]) if t not in(puncts)]
        #words = [re.sub('\d','#',t) for t in word_tokenize(sentence[x]) if t not in(puncts)]
        sentences.append(words)
    doc = sent_tokenize(df_emb['non-formal'][y])
    sentence=[sent for sent in doc]
    for x in range(len(sentence)):
        words = [t for t in word_tokenize(sentence[x]) if t not in(puncts)]
        #words = [re.sub('\d','#',t) for t in word_tokenize(sentence[x]) if t not in(puncts)]
        sentences.append(words)

print(sentences)

from gensim.models import Word2Vec

model1 = Word2Vec(sentences, sg=1, size=EMBEDDING_SIZE, window=5, min_count=1, workers=4)

model1.wv.most_similar('pembayaran')

#model.wv.save_word2vec_format('./hasil_wordembedding/wordembedding.bin')
model1.wv.save_word2vec_format('wordembedding100.txt',binary=False)

"""# Embedding"""

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

wordembedding_file = open(r'/content/wordembedding100.txt', encoding="utf8")

for line in wordembedding_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
wordembedding_file.close()

"""Ingatlah bahwa kami memiliki 3331 kata unik dalam input. Kami akan membuat matriks di mana nomor baris akan mewakili nilai integer untuk kata dan kolom akan sesuai dengan dimensi kata. Matriks ini akan berisi kata embeddings untuk kata-kata dalam kalimat input kami."""

num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
for word, index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

"""Pertama-tama mari kita cetak kata embeddings untuk kata "rino" menggunakan kamus embedding yang telah dibuat."""

print(embeddings_dictionary["pembayaran"])

"""Pada bagian sebelumnya, kita melihat bahwa representasi integer untuk kata "rino" adalah 1107. Sekarang mari kita periksa indeks ke-1107 dari kata embedding matrix."""

print(embedding_matrix[1107])

"""Anda dapat melihat bahwa nilai-nilai untuk baris ke-1107 dalam matriks penyematan mirip dengan representasi vektor dari kata "rino" dalam kamus embedding word yang telah dibuat, yang mengonfirmasi bahwa baris dalam matriks penyematan mewakili embeddings kata yang sesuai dari kamus penyematan embedding word. Matriks embedding kata ini akan digunakan untuk membuat layer embedding untuk model LSTM kami.

Script berikut ini membuat layer embedding untuk input:
"""

embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len)

"""#Membuat Model

Sekarang saatnya untuk mengembangkan model kami. Hal pertama yang perlu kita lakukan adalah mendefinisikan output kita, karena kita tahu bahwa output akan menjadi urutan kata-kata. 

Ingat bahwa jumlah total kata unik dalam output adalah 3179 + 1<end>. 

Oleh karena itu, setiap kata dalam output dapat berupa salah satu dari 3180 kata. Panjang kalimat output adalah 30. 

Dan untuk setiap kalimat input, kita membutuhkan kalimat output yang sesuai. Oleh karena itu, bentuk akhir dari output adalah:



```
(number of inputs, length of the output sentence, the number of words in the output)
```

Script berikut ini membuat array output kosong:
"""

decoder_targets_one_hot = np.zeros((
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)

decoder_targets_one_hot.shape

decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')

"""Untuk membuat prediksi, lapisan akhir model akan menjadi *dense layer*, oleh karena itu kita perlu output dalam bentuk one-hot encoded vectors, karena kita akan menggunakan fungsi aktivasi softmax pada *dense layer*. 

Untuk membuat output one-hot encoded seperti itu, langkah selanjutnya adalah menetapkan 1 ke nomor kolom yang sesuai dengan representasi bilangan bulat kata tersebut. 

Misalnya, representasi integer untuk <sos> je suis malade adalah [2 3 6 188 0 0 0 0 0 0 0]. 

Dalam larik keluaran decoder_targets_one_hot, di kolom kedua dari baris pertama, 1 akan dimasukkan. Demikian pula, pada indeks ketiga dari baris kedua, 1 lainnya akan dimasukkan, dan seterusnya.
"""

for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1

encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]

"""Langkah selanjutnya adalah mendefinisikan decoder. Dekoder akan memiliki dua input: keadaan tersembunyi dan keadaan sel dari encoder dan kalimat input, yang sebenarnya akan menjadi kalimat keluaran dengan token <sos> yang ditambahkan di awal.

Script berikut membuat decoder LSTM:
"""

decoder_inputs_placeholder = Input(shape=(max_out_len,))

decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

"""Akhirnya, output dari LSTM decoder dilewatkan melalui lapisan padat untuk memprediksi output decoder, seperti yang ditunjukkan di sini:"""

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

"""```
keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```
"""

model = Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder], decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from keras.utils import plot_model
plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

"""Dari output, Anda dapat melihat bahwa kami memiliki dua jenis input. input_1 adalah tempat penampung input untuk encoder, yang tertanam dan melewati lapisan lstm_1, yang pada dasarnya adalah LSTM pembuat enkode. Ada tiga output dari lapisan lstm_1: output, layer tersembunyi dan status sel. Namun, hanya keadaan sel dan keadaan tersembunyi dilewatkan ke decoder.

Di sini lapisan lstm_2 adalah LSTM decoder. Input_2 berisi kalimat keluaran dengan token <sos> ditambahkan di awal. Input_2 juga dilewatkan melalui lapisan embedding dan digunakan sebagai input ke LSTM decoder, lstm_2. Akhirnya, output dari LSTM decoder dilewatkan melalui lapisan padat untuk membuat prediksi.

Langkah selanjutnya adalah melatih model menggunakan metode fit ():
"""

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# from keras.callbacks import ModelCheckpoint
# # define the checkpoint
# filepath = "Model100.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

r = model.fit(
    [encoder_input_sequences, decoder_input_sequences],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1
)

"""Saat berlatih, kita tahu input aktual ke decoder untuk semua kata output dalam urutan. 

Anda dapat melihat bahwa input ke decoder dan output dari decoder diketahui dan model dilatih berdasarkan input dan output ini.

Namun, selama prediksi kata berikutnya akan diprediksi berdasarkan kata sebelumnya, yang pada gilirannya juga diprediksi pada langkah waktu sebelumnya. Sekarang Anda akan memahami tujuan token <sos> dan <end>. Saat membuat prediksi aktual, urutan output penuh tidak tersedia, pada kenyataannya itulah yang harus kami prediksi. Selama prediksi, satu-satunya kata yang tersedia bagi kami adalah <sos> karena semua kalimat keluaran dimulai dengan <sos>.
"""

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

"""#Load Model"""

# from keras.models import load_model
# # load model
# model = load_model('/content/drive/My Drive/Colab Notebooks/Skripsi/Model/Model100.h5')
# model.load_weights("/content/drive/My Drive/Colab Notebooks/Skripsi/Model/Model100.h5")

# model.compile(
#     optimizer='rmsprop',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# import matplotlib.pyplot as plt

# # Plot training & validation accuracy values
# plt.plot(model.history['accuracy'])
# plt.plot(model.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(model.history['loss'])
# plt.plot(model.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

"""Pada langkah 1, keadaan tersembunyi dan keadaan sel encoder, dan <sos>, digunakan sebagai input ke decoder. Dekoder memprediksi kata y1 yang mungkin benar atau tidak. Namun, sesuai model kami, probabilitas prediksi yang benar adalah 0,7911. Pada langkah 2, keadaan tersembunyi decoder dan keadaan sel dari langkah 1, bersama dengan y1, digunakan sebagai input ke dekoder, yang memprediksi y2. Proses berlanjut hingga token <eos> ditemukan. Semua output yang diprediksi dari decoder kemudian digabungkan untuk membentuk kalimat output akhir. Mari kita modifikasi model kita untuk mengimplementasikan logika ini."""

encoder_model = Model(encoder_inputs_placeholder, encoder_states)

encoder_model.save('encoder_model300.h5')  # creates a HDF5 file 'my_model.h5'

"""Model encoder tetap sama

Karena sekarang pada setiap langkah kita membutuhkan dekoder hidden dan cell state, kita akan memodifikasi model kita untuk menerima keadaan tersembunyi dan sel seperti yang ditunjukkan di bawah ini:
"""

decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

"""Sekarang pada setiap langkah waktu, hanya akan ada satu kata dalam input decoder, kita perlu memodifikasi layer embedding decoder sebagai berikut:"""

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

"""Selanjutnya, kita perlu membuat placeholder untuk output decoder:"""

decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

"""Untuk membuat prediksi, output decoder dilewatkan melalui dense layer:"""

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)

"""Langkah terakhir adalah menentukan model dekoder yang diperbarui, seperti yang ditunjukkan di sini"""

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

decoder_model.save('decoder_model300.h5')  # creates a HDF5 file 'my_model.h5'

from keras.utils import plot_model
plot_model(decoder_model, to_file='model_plot_dec.png', show_shapes=True, show_layer_names=True)

"""Pada gambar di atas, lstm_2 adalah dekoder LSTM yang dimodifikasi. Anda dapat melihat bahwa ia menerima kalimat dengan satu kata seperti yang ditunjukkan pada input_5, dan status sel dan tersembunyi dari output sebelumnya (input_3 dan input_4). Anda dapat melihat bahwa bentuk kalimat input sekarang (tidak ada, 1) karena hanya akan ada satu kata dalam input dekoder. Sebaliknya, selama pelatihan bentuk kalimat input adalah (Tidak ada, 6) karena input berisi kalimat lengkap dengan panjang maksimum 6.

#Membuat Prediksi

Pada langkah ini, Anda akan melihat bagaimana membuat prediksi menggunakan kalimat non-formal sebagai input.

Pada langkah-langkah tokenization, kami mengonversi kata menjadi bilangan bulat. Output dari decoder juga akan menjadi bilangan bulat. Namun, kami ingin hasil kami menjadi urutan kata dalam kalimat formal. Untuk melakukannya, kita perlu mengonversi bilangan bulat kembali ke kata-kata. Kami akan membuat kamus baru untuk input dan output di mana kunci akan menjadi bilangan bulat dan nilai yang sesuai akan menjadi kata-kata.
"""

idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}

# Save
np.save('idx2word_input.npy', idx2word_input)

# Save
np.save('idx2word_target.npy', idx2word_target)

"""Selanjutnya kita akan membuat metode, yaitu translate_sentence (). Metode ini akan menerima urutan kalimat bahasa Inggris input-padded (dalam bentuk bilangan bulat) dan akan mengembalikan kalimat bahasa Prancis yang diterjemahkan. Lihatlah metode translate_sentence ():"""

def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<end>']
    output_sentence = []

    for _ in range(max_out_len):
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

    return ' '.join(output_sentence)

i=120
print(input_sentences[i])
input_seq = encoder_input_sequences[i:i+1]
states_value = encoder_model.predict(input_seq)
target_seq = np.zeros((1, 1))
target_seq[0, 0] = word2idx_outputs['<sos>']
eos = word2idx_outputs['<end>']

output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
print('output_tokens: ', output_tokens)
print(len(output_tokens[0, 0, :]))
print(idx2word_target[np.argmax(output_tokens[0, 0, :])])

"""Dalam skrip di atas kita meneruskan urutan input ke encoder_model, yang memprediksi keadaan tersembunyi dan keadaan sel, yang disimpan dalam variabel state_value.

* Selanjutnya, kita mendefinisikan variabel target_seq, yang merupakan matriks 1 x 1 dari semua nol. 
* Variabel target_seq berisi kata pertama ke model decoder, yaitu "sos"

* Setelah itu, variabel end diinisialisasi, yang menyimpan nilai integer untuk token "end". Pada baris berikutnya, daftar output_sentence didefinisikan, yang akan berisi terjemahan yang diprediksi.

* Selanjutnya, kita jalankan for. Jumlah siklus eksekusi untuk for loop sama dengan panjang kalimat terpanjang dalam output. 

* Di dalam loop, dalam iterasi pertama, decoder_model memprediksi output dan status sel dan tersembunyi, menggunakan status sel dan tersembunyi dari enkoder, dan token input, mis. <sos>. 
* Indeks kata yang diprediksi disimpan dalam variabel idx. Jika nilai indeks prediksi sama dengan token <eos>, loop berakhir. Lain jika indeks prediksi lebih besar dari nol, kata yang sesuai diambil dari kamus idx2word dan disimpan dalam variabel kata, yang kemudian ditambahkan ke daftar output_sentence. 
* Variabel States_value diperbarui dengan keadaan sel dan tersembunyi baru dari dekoder dan indeks kata yang diprediksi disimpan dalam variabel target_seq. 
* Dalam siklus loop berikutnya, status sel dan tersembunyi yang diperbarui, bersama dengan indeks kata yang diprediksi sebelumnya, digunakan untuk membuat prediksi baru. 
* Loop berlanjut hingga panjang urutan output maksimum tercapai atau token <eos> ditemukan.



Akhirnya, kata-kata dalam daftar output_sentence disatukan menggunakan spasi dan string yang dihasilkan dikembalikan ke fungsi pemanggilan.

#Testing Model

Untuk menguji kode, kita akan secara acak memilih kalimat dari daftar input_sentences, mengambil urutan padded yang sesuai untuk kalimat, dan akan meneruskannya ke metode translate_sentence ().
"""

encoder_input_sequences.shape

# i = np.random.choice(len(input_sentences))
i=120
print(input_sentences[i])
input_seq = encoder_input_sequences[i:i+1]
input_seq

# i = np.random.choice(len(input_sentences))
input_seq = encoder_input_sequences[i:i+1]
terjemahan = translate_sentence(input_seq)
print('Input:', input_sentences[i])
print('Response:', terjemahan)

"""#Kesimpulan dan Perspektif

Terjemahan mesin saraf adalah aplikasi yang cukup maju dari pemrosesan bahasa alami dan melibatkan arsitektur yang sangat kompleks.

Artikel ini menjelaskan cara melakukan terjemahan mesin saraf melalui arsitektur seq2seq, yang pada gilirannya didasarkan pada model encoder-decoder. Encoder adalah LSTM yang mengkodekan kalimat input sedangkan decoder menerjemahkan input dan menghasilkan output yang sesuai. Teknik yang dijelaskan dalam artikel ini dapat digunakan untuk membuat model terjemahan mesin apa pun, selama dataset dalam format yang mirip dengan yang digunakan dalam artikel ini. Anda juga dapat menggunakan arsitektur seq2seq untuk mengembangkan chatbots.

Arsitektur seq2seq cukup berhasil ketika memetakan relasi input ke output. Namun, ada satu batasan untuk arsitektur seq2seq. Arsitektur vanilla seq2seq yang dijelaskan dalam artikel ini tidak mampu menangkap konteks. Ini hanya belajar memetakan input mandiri ke output mandiri. Percakapan real-time didasarkan pada konteks dan dialog antara dua atau lebih pengguna didasarkan pada apa pun yang dikatakan di masa lalu. Oleh karena itu, model seq2seq berbasis encoder-decoder sederhana tidak boleh digunakan jika Anda ingin membuat chatbot yang cukup canggih.

#Testing Kalimat Inputan
"""

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
  list_bantu = [x for x in range(max_input_len)]

  # array untuk encoder
  jadi = []
  jadi.append(list_bantu)
  jadi.append(test_word2idx)

  encoder_input_sequences_test = pad_sequences(jadi, maxlen=28)

  return encoder_input_sequences_test[1:2]

"""#Data Testing"""

worksheet = gc.open('Dataset Kalimat').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()
# Convert to a DataFrame and render.
import pandas as pd
colsname=['formal','non-formal']
df_test = pd.DataFrame.from_records(rows,columns=colsname)
df_test = df_test[3000:3100]

df_test = df_test.reset_index(drop=True)
df_test

def input_test(w):
  w = w.lower()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = w.rstrip().strip()

  return w

input_sentences_test = []
for x in range(100):
  input_sentences_test.append(input_test(df_test['non-formal'][x]))

#data testing
for x in range(200):
  input_teks = input_sentences_test[x]
  input_seq = preproces(input_teks)
  terjemahan = translate_sentence(input_seq)
  print('Input:', input_teks)
  print('Response:', terjemahan)
  print('========================================================')

"""#Kalimat INputan"""

input_teks = "dia nggak bisa"
input_seq = preproces(input_teks)
terjemahan = translate_sentence(input_seq)
print('Input:', input_teks)
print('Response:', terjemahan)

