from keras.models import Sequential
from keras import layers
from keras.utils import np_utils
import numpy as np
from keras.utils import to_categorical


class LanguageModel:
    def __init__(self, ngram_size, tokenizer):
        self.ngram_size = ngram_size
        self.tokenizer = tokenizer
        self.model = Sequential()
        # TODO: Implement
        self.mu = 0
        self.U_k = None
        self.eigvals = None

    def train(self, train_X, epochs=10, batch_size=8):
        # TODO: Implement

        predictors, label = train_X[:,:-1], train_X[:,-1]

        self.model.add(layers.Dense(10, input_dim=train_X.shape[1]-1, activation='tanh'))
        self.model.add(layers.Dense(20, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics='accuracy')
        self.model.summary()

        history = self.model.fit(predictors, label, epochs=epochs, batch_size=batch_size)

        return history

    def predict(self, context):
        # TODO: Implement
        logits = []
        pred_index = 0
        return pred_index, logits

    def generate(self, context, max_num_words=20):
        output = []
        # TODO: Implement

        return output

    def sent_log_likelihood(self, ngrams):
        logprob = 0
        # TODO: Implement

        return logprob

    def fill_in(self, prefix, suffix, get_ngrams_fn):
        # TODO: Implement (MSc Students only)
        logits = []
        pred_word_id = 0
        return pred_word_id, logits

    def get_word_embedding(self, word):
        return self.model.layers[0].get_weights()[0][self.tokenizer.word_index[word]]
