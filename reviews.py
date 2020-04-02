from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",    # get IMBD reviews dataset already in preprocessed form
                                                      num_words=5000,     #consider only top 5000 most frequent words
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
print(x_train.shape, y_train.shape)

x_train = pad_sequences(x_train, maxlen = 500)  # all sequences are of equal length (500)
x_test = pad_sequences(x_test, maxlen = 500)

print(x_train.shape)

y_train = to_categorical(y_train)

els = [max(x) for x in x_train]
print(max(els)) # size of vocabulary

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(5000, embed_dim, input_length = 500))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())
model.fit(x_train, y_train, epochs = 1 ,verbose = 1)

score,acc = model.evaluate(x_test, y_test, verbose = 1)
print("Score: %.2f" % (score))
print("Validation Accuracy: %.2f" % (acc))

# Now that the model is ready we can check our own text and see if it will be matched correctly by model

from keras.preprocessing.text import text_to_word_sequence

word_index=imdb.get_word_index()

text = "The film was terrible. I hated it. Acting was not good and the story was boring"
words = text_to_word_sequence(text)
translate = np.array([word_index[word] if word in word_index else 0 for word in words])
print(translate)

pr = np.zeros((500 - len(translate),), dtype=int)
pr = np.append(pr, translate)
if (pr.ndim == 1):
    pr = np.array([pr])
print(pr.shape)

res = model.predict(pr)
print(res) # outputs probability of review to be neg(0) or pos(1), in my case looks like this [[0.9393943  0.06060567]]
