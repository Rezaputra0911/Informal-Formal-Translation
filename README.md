# Informal-Formal-Translation
natural language processing for machine translation of informal sentences into formal Indonesian sentences

![alt text](https://raw.githubusercontent.com/Rezaputra0911/Informal-Formal-Translation/master/static/img/Untitled.png)
The high frequency of using formal sentences does not guarantee that Indonesians are fluent in using formal sentences, especially in determining the standard words to be used. The number of foreign languages that you want to master, the imbalance in the use of standard and non-standard sentences, ignoring the definition of a sentence, and low mastery of language structures are factors in the difficulty of determining standard sentences in writing

# Introduction
Through this research, problems in changing non-standard sentences can be studied using the Natural Language Processing method and the Long Short Term Memory machine learning algorithm. The writing of informal sentences is studied to be translated into formal sentences by paying attention to the order of words in a sentence. The sentence that has been inputted will be broken down into several words using the Tokenization method. The pre-processed data in the form of a vector from the pad_sqeunces process becomes an input for the Long Short Term Memory machine learning algorithm. The machine translation process is also carried out with
encoder-decoder process.

The encoder is used to encode an arrangement of sentences into input in the prediction process. Meanwhile, a decoder is used to restore the code into a predictive sentence. To test the accuracy of predictive formal sentences, the researcher used the BLEU evaluation to obtain the evaluation value of the sentence prediction results. The evaluation compares the predicted n-grams with the translated n-grams of the reference from the dataset and calculates the number of similarities.

## Formal and Informal Sentences in Indonesian
Formal sentences are sentences that are arranged based on good and correct Indonesian. Because they are composed in an understanding with the rules of the Indonesian language, formal sentences have standard words in them. In addition, formal sentences have the following characteristics: they contain sentence elements in Indonesian, and have propositions in them, the use of function words or conjunctions and complements that are used effectively. Serves as a sentence to convey or express logical reasoning, and formal sentences have a complete idea.

An informal sentence or what is called a non-formal sentence is a sentence that contradicts the rules of language structure. Non-formal sentences are used in day-to-day discussions among peers. Non-formal sentences have the following characteristics: additional affixes, word changes occur, interference occurs in the sentences used, word shortening occurs

| **_Bentukan Kata Formal_** | **_Bentukan Kata Informal_**|
| ------------- | ------------- |
| Berkata  | Bilang  |
| Hanya  | Cuma  |
| Lepas | Copot  |
| Tidak  | Nggak  |
| Mencuci| Nyuci  |
| Ditingkatkan | Ditingkatin  |

# Installing

it should be noted, the application was developed with python 3. You need to install python 3.
some required python libraries:
* [Tensorflow](https://www.tensorflow.org/install) - Prediction Model Development
* [Keras](https://keras.io/guides/) - Prediction Model Development
* [NLTK](http://www.nltk.org/data.html?highlight=proxy) - Natural Language Processing
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) - access the dataset easily
* [Flask](https://flask.palletsprojects.com/en/1.1.x/installation/) - Web Application with python Framework



# Running the tests

to run model development, running file [model traning.py](https://github.com/Rezaputra0911/Informal-Formal-Translation/blob/master/model_training.py) for getting **model**,**idx2word**, and **word2idx**. running for encoder-decoder model with embedding matrix weights of 100,200, and 300.

![alt text](https://raw.githubusercontent.com/Rezaputra0911/Informal-Formal-Translation/master/static/img/encoder.png)
![alt text](https://raw.githubusercontent.com/Rezaputra0911/Informal-Formal-Translation/master/static/img/decoder.png)

for testing using the BLEU score, running file [score_evaluasi_bleu.py](https://github.com/Rezaputra0911/Informal-Formal-Translation/blob/master/score_evaluasi_bleu.py). 

![alt text](https://raw.githubusercontent.com/Rezaputra0911/Informal-Formal-Translation/master/static/img/sentences%20bleu.png)
![alt text](https://raw.githubusercontent.com/Rezaputra0911/Informal-Formal-Translation/master/static/img/corpus.png)


# Deployment

For system deployment, I use a website with the Flask framework to connect HTML with the python model.

To run a website based program, run [app.py](https://github.com/Rezaputra0911/Informal-Formal-Translation/blob/master/app.py).


# Open Source Dataset
Since I can't find any open source or corpus data sets that translate non-standard sentences into standard sentences, I release my own dataset to download and use for free. Currently, I have around 3000 Indonesian sentences in my data set and more are being added. About 300 000 formal sentences that can be translated into informal sentences. These sentences are obtained through the process of crawling and scrapping one of the news portal sites.


# Acknowledgments

Special Thanks for to my mentor Yulius Deni!
