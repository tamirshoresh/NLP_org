########################################################################################################################
#library imports:
########################################################################################################################
import os
import math
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf #"tensorflow>=1.15,<2.0"
import tensorflow_hub as hub
import spacy
import pickle
from tqdm.notebook import tqdm
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
#%matplotlib inline
#%config InlineBackend.figure_format='retina'
#pd.set_option('display.max_colwidth', 200)

########################################################################################################################
#importing IMDb database:
########################################################################################################################
# get reproducible results
from numpy.random import seed
seed(0xdeadbeef)
import tensorflow as tf
tf.random.set_seed(0xdeadbeef)
#actual importation
imdb = keras.datasets.imdb
num_words = 20000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(seed=1, num_words=num_words)

print('\nExample of data sequence and corresponding label:')
print('data sequence:', train_data[0])
print('label:', train_labels[0])
print('\n')

'''
We see that the text of the review has been encoded as a sequence of integers. Please refer to part 2 of this tutorial series if you want to understand how such an encoding can be done in practice.
Each word in the text is represented as an integer. A dictionary called the vocabulary links each word to a unique integer.
In the example above, we see that the integer 4 is repeated many times. This integer corresponds to a very frequent word.
And actually, the more frequent a word, the lower the integer.
To decode the review, we need to make use of the vocabulary:
'''

# A dictionary mapping words to an integer index
vocabulary = imdb.get_word_index()

# The first indices are reserved
vocabulary = {k:(v+3) for k,v in vocabulary.items()}
vocabulary["<PAD>"] = 0
# See how integer 1 appears first in the review above.
vocabulary["<START>"] = 1
vocabulary["<UNK>"] = 2  # unknown
vocabulary["<UNUSED>"] = 3

# reversing the vocabulary.
# in the index, the key is an integer,
# and the value is the corresponding word.
index = dict([(value, key) for (key, value) in vocabulary.items()])

def decode_review(text):
    '''converts encoded text to human readable form.
    each integer in the text is looked up in the index, and
    replaced by the corresponding word.
    '''
    return ' '.join([index.get(i, '?') for i in text])

print('\nExample of decoded sequence:\n')
print(decode_review(train_data[0]))
print('\n')

'''
We see that integer 4 indeed corresponds to a very frequent word, "the".
Now what do we do with this dataset? We can see two issues if we are to use it as input to a neural network:
    1. The reviews have a variable number of words, while the network has a fixed number of neurons.
    2. The words are completely independent.
        For example, "brilliant" and "awesome" correspond to two different integers, and the neural network does not know a priori that these two adjectives have similar meaning.
Let's deal with the first issue.
To get a fixed length input, we can simply truncate the reviews to a fixed number of words, say 256.
For reviews that have more than 256 words, we will keep only the first 256 words. For shorter reviews, we will fill the unused word slots with zeros.
With keras, this is easy to do:
'''

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=vocabulary["<PAD>"],
                                                        padding='post',
                                                        maxlen=196)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=vocabulary["<PAD>"],
                                                       padding='post',
                                                       maxlen=196)
print('\nExample of decoded and padded sequence:\n')
print(decode_review(train_data[0]))
print('\n')

'''
the first issue is resolved.
The second issue will be resolved using the embedding og BERT and ELMo. To get to that part we need to create the df using the 'decode_review' function
'''

train_df = pd.DataFrame(columns = ['label', 'review'])
test_df = pd.DataFrame(columns = ['label', 'review'])

for n in range(len(train_data)):
  d_train = {'label': train_labels[n],
             'review': decode_review(train_data[n])}
  d_test = {'label': test_labels[n],
            'review': decode_review(test_data[n])}
  train_df = train_df.append(d_train, ignore_index = True)
  test_df = test_df.append(d_test, ignore_index = True)

print('\nHead of the training dataset:\n')
print(train_df.head())
print('\n')
print('Head of the testing dataset:\n')
print(test_df.head())
print('\n')

#now we need to process the data to fit the ELMo model
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

train_df['review'] = train_df['review'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
test_df['review'] = test_df['review'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
train_df['review'] = train_df['review'].str.lower()
test_df['review'] = test_df['review'].str.lower()

# remove numbers
train_df['review'] = train_df['review'].str.replace("[0-9]", " ")
test_df['review'] = test_df['review'].str.replace("[0-9]", " ")

# remove whitespaces
train_df['review'] = train_df['review'].apply(lambda x:' '.join(x.split()))
test_df['review'] = test_df['review'].apply(lambda x: ' '.join(x.split()))
########################################################################################################
#import spacy for ELMo:
#########################################################################################################
# import spaCy's language model
nlp = spacy.load('en', disable=['parser', 'ner'])

# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

train_df['review'] = lemmatization(train_df['review'])
test_df['review'] = lemmatization(test_df['review'])
print('head of train_df:')
print(train_df.head())
print('head of test_df:')
print(test_df.head())

#create elmo
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
#example with x
x = ["Amazing! Loved this so much! Very Happy."]
embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
print('elmo embedding shape: '+str(embeddings.shape))

def elmo_vectors(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

list_train = [train_df[i:i+100] for i in range(0,train_df.shape[0],100)]
list_test = [test_df[i:i+100] for i in range(0,test_df.shape[0],100)]

elmo_train = [elmo_vectors(x['review']) for x in list_train]
elmo_test = [elmo_vectors(x['review']) for x in list_test]

elmo_train_new = np.concatenate(elmo_train, axis = 0)
elmo_test_new = np.concatenate(elmo_test, axis = 0)

# save elmo_train_new
pickle_out = open(r"C:\Users\tamir\Desktop\ELMo\elmo_train_03032019.pickle","wb")
pickle.dump(elmo_train_new, pickle_out)
pickle_out.close()

# save elmo_test_new
pickle_out = open(r"C:\Users\tamir\Desktop\ELMo\elmo_test_03032019.pickle","wb")
pickle.dump(elmo_test_new, pickle_out)
pickle_out.close()

# load elmo_train_new
pickle_in = open(r"C:\Users\tamir\Desktop\ELMo\elmo_train_03032019.pickle", "rb")
elmo_train_new = pickle.load(pickle_in)

# load elmo_train_new
pickle_in = open(r"C:\Users\tamir\Desktop\ELMo\elmo_test_03032019.pickle", "rb")
elmo_test_new = pickle.load(pickle_in)

#training a full model
from sklearn.model_selection import train_test_split
#creating a validation set from the training set
xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new,
                                                  train['rating'],
                                                  random_state=42,
                                                  test_size=0.2)

#creating the actual sentiment analysis model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense,Dropout, Lambda)
from tensorflow.keras.layers import Input

def ELMoEmbedding(x):
  return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def build_model():
  input_text = Input(shape=(1,), dtype="string")
  x = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
  x = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
  x = Dense(2, activation="sigmoid")(x)
  model = Model(inputs=[input_text], outputs=x)
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  return model

model = build_model()
model.summary()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  history = model.fit(x=xtrain, y=ytrain, batch_size=50, epochs=3, validation_split=0.2)

########################################################################################################

#plots from ELMo
#%load_ext tensorboard
#%tensorboard --logdir log

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'])
plt.title('Loss over training epochs')
plt.figure(figsize=(12,8))
plt.show();

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'])
plt.title('Accuracy over training epochs')
plt.figure(figsize=(12,8))
plt.show();

########################################################################################################################
#prediction using ELMo
########################################################################################################################
# Predict using argmax to ensure we have a single label outputted (highest probability)
y_pred = model.predict(X_test).argmax(axis=-1)

# Ratings start from 0 in this model so we need to use this array for the class labels
labels_cm = [0,1]
cm = confusion_matrix(y_test, y_pred, labels=labels_cm)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)

hmap = sns.heatmap(df_cm, annot=True, fmt="d")
hmap.yaxis.set_ticklabels([0,1], ha='right')
hmap.xaxis.set_ticklabels([0,1], ha='right')

plt.ylabel('True label')
plt.xlabel('Predicted label');
plt.savefig(r'C:\Users\tamir\Documents\תואר שני\afeka\NLP\final project\plots\Confusion Matrix.png', bbox_inches="tight")


# Test sentences to use.
sentences = ["This movie is rubbish, would not recommend.",
             "Amazing! Loved this so much! Great film.",
             "Very average...",
             "I probably wouldn't go see it again, but overall i guess it was alright",
             "Not a big fan"]

# Pre-process the test strings like training/test data has done.
pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
pred_token_ids = map(lambda tids: tids +[0]*(max_seq_length-len(tids)),pred_token_ids)
pred_tokens = map(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)), pred_tokens)
pred_tokens = pred_tokens.str.lower()
pred_tokens = pred_tokens.str.replace("[0-9]", " ")
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict(pred_token_ids).argmax(axis=-1)

for text, label in zip(sentences, predictions):
    print("\ntext: ", text)
    print("rating: ", classes[label])
    print()
