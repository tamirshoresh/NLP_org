########################################################################################################################
#library imports:
########################################################################################################################
import os
import math
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import bert
from tqdm.notebook import tqdm
from tensorflow import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

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

'''
We can now compare how many reviews we have in the training set.
We will see that the positive (1) and negative (0) reviews are balanced at 12500 samples each
'''

# Outputs the number of reviews by review type to understand the spread of review labels in the set.
sns.set(font_scale=2, rc={'figure.figsize':(10,5)})
sns.set_palette(sns.color_palette(["#FF0000", "#FFA500"]))#, "#FFFF00", "#00FFFF", "#00FF00"]))
chart = sns.countplot(train_df['label'])
plt.title("reviews per label")
chart.set_xticklabels(chart.get_xticklabels(), horizontalalignment='right');
chart.set(xlabel= "Labels", ylabel="Number of Reviews")
plt.savefig(r'C:\Users\tamir\Documents\תואר שני\afeka\NLP\final project\plots\train_data_distribution.png', bbox_inches="tight")

total_reviews = len(train_df['label'])
print("Total Reviews: ", total_reviews)

########################################################################################################################
#Unpacking BERT and creating the model:
########################################################################################################################

# Retreive the Uncased BERT-Base model.
#run the next command in terminal:
#wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

import os
bert_model_name = "uncased_L-12_H-768_A-12"
bert_ckpt_dir = r"C:\Users\tamir\Documents\תואר שני\afeka\NLP\final project\bert\uncased_L-12_H-768_A-12"
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")



# Called on the un-processed dataset.
def get_data(train, test, tokenizer: FullTokenizer, classes, max_seq_length_override=256):
    # Finds the max sequence length of the input data.
    max_seq_length = 0
    print("Processing training data...")
    train_x, train_y, max_seq_length = process_data(train, max_seq_length)
    print("Processing testing data...")
    test_x, test_y, max_seq_length = process_data(test, max_seq_length)

    # Ensure the overriden value is only used if the sequence length of the dataset exceeds it.
    max_seq_length = min(max_seq_length, max_seq_length_override)

    # Pads the token ids from the previous step
    train_x = pad_data(train_x, max_seq_length)
    test_x = pad_data(test_x, max_seq_length)

    return ((train_x, train_y), (test_x, test_y)), max_seq_length

def process_data(dataframe, max_seq_length):
    x, y = [], []
    # Progress bar used when processing a large dataset.
    #with tqdm(total=total_reviews) as pbar:
    for _, row in dataframe.iterrows():
        text, label = row["review"], row["label"]
        #Tokenize the x labels
        tokens = tokenizer.tokenize(text)
        # Prepend [CLS] and append [SEP] to the tokenized x.
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # Convert the tokens to ids.
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Update max sequence length if it is a new max value.
        max_seq_length = max(max_seq_length, len(token_ids))
        x.append(token_ids)
        y.append(classes.index(label))
        #pbar.update(1)
    return np.array(x), np.array(y), max_seq_length

def pad_data(ids, max_seq_length):
    x_new = []
    for input_ids in ids:
        print(input_ids)
        input_ids = input_ids[:min(len(input_ids), max_seq_length - 2)]
        print(input_ids)
        input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
        print(input_ids)
        x_new.append(np.array(input_ids))
    return np.array(x_new)


def create_model(max_seq_length, bert_ckpt_file):
    # Read in bert configuration from file and apply it to bert model layer
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    # Flattens the middle dimension (max sequence length) input layer, ignores the first and last dimension. Eg. (None, 196, 768) -> (None, 768).
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    # Dropout used to prevent overfitting.
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    # 768 used as this is the default hidden size set in the config.
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    # Second dropout layer.
    logits = keras.layers.Dropout(0.5)(logits)
    # Output layer, softmax used as we want classification. Outputs probabilities for each rating that all add up to 1.
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    # Load model.
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_length))

    # Load the pre-trained weights so we can fine-tune the model to the dataset.
    load_stock_weights(bert, bert_ckpt_file)
    return model

#tokenizing before BERT
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
# labels 0 or 1 stored in a list.
classes = train_df['label'].unique().tolist()
((X_train, y_train), (X_test, y_test)), max_seq_length = get_data(train_df, test_df, tokenizer, classes, max_seq_length_override=196);

#creating the model
model = create_model(max_seq_length, bert_ckpt_file)
print(model.summary())
model.compile(
    # Adam recommended for BERT, if we are getting OOM errors then we should try a different optimizer.
    optimizer=keras.optimizers.Adam(2e-5),
    # Sparse is used instead of normal categorical cross entropy due to integers being used instead of one-hot encoding.
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

#fit model
#log_dir = "log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
history = model.fit(x=X_train,
                    y=y_train,
                    validation_split=0.25,
                    batch_size=50,
                    shuffle=True,
                    epochs=3)#,
                 #   callbacks=[tensorboard_callback])
# Save the pre-trained weights to use when predicting.
model.save_weights(r'C:\Users\tamir\Documents\תואר שני\afeka\NLP\final project\bert\after pretrain\bert_weights.h5')
print('\nmodel history:\n')
print(history.history)

#plots from BERT
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
plt.savefig(r'C:\Users\tamir\Documents\תואר שני\afeka\NLP\final project\plots\Loss over training epochs.png', bbox_inches="tight")

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
plt.savefig(r'C:\Users\tamir\Documents\תואר שני\afeka\NLP\final project\plots\Accuracy over training epochs.png', bbox_inches="tight")

########################################################################################################################
#prediction using BERT
########################################################################################################################
# Load the weights to ensure it is definitely using our fine-tuned model
model.load_weights(r'C:\Users\tamir\Documents\תואר שני\afeka\NLP\final project\bert\after pretrain\bert_weights.h5')
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
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict(pred_token_ids).argmax(axis=-1)

for text, label in zip(sentences, predictions):
    print("\ntext: ", text)
    print("rating: ", classes[label])
    print()
