import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.models import load_model
from tqdm import tqdm


class WineClassifier:
    def __init__(self, vocabulary_size, embedding_size, max_seq_length, embedding_weights, num_classes):
        VOCABULARY_SIZE = vocabulary_size
        EMBEDDING_SIZE = embedding_size
        MAX_SEQ_LENGTH = max_seq_length
        embedding_weights = embedding_weights
        NUM_CLASSES = num_classes

        import tensorflow.python.keras.activations as a
        self._model = tf.keras.Sequential([
            Embedding(
                input_dim=VOCABULARY_SIZE,
                output_dim=EMBEDDING_SIZE,
                input_length=MAX_SEQ_LENGTH,
                weights=[embedding_weights],
                trainable=True
            ),
            L.Bidirectional(L.LSTM(64, return_sequences=False)),
            L.Dense(64, activation=a.relu),
            L.Dense(32, activation=a.relu),
            L.Dense(32, activation=a.relu),
            L.Dense(16, activation=a.relu),
            L.Dense(NUM_CLASSES, activation=a.softmax)
        ])

        self._model.compile(
            loss=SparseCategoricalCrossentropy(),
            optimizer='adam',
            metrics=['accuracy']
        )
        self._results = None
        # self._model.summary()

    @property
    def model(self):
        return self._model

    def fit(self, X_train, y_train, epochs, batch_size, X_validation, y_validation):
        early_stopping = EarlyStopping()
        self._results = self._model.fit(
            X_train.reshape(*X_train.shape, 1),
            np.array(y_train),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_validation.reshape(*X_validation.shape, 1), np.array(y_validation)),
            callbacks=[early_stopping]
        )

    def load(self, load_path: str):
        self._model = load_model(load_path)

    def save(self, save_path: str):
        self._model.save(save_path)

    def evaluate(self, X_test, y_test):
        self._model.evaluate(
            X_test,
            y_test
        )

    def predict(self, X_test):
        return self._model.predict_classes(X_test)


def emb_weights(word2vec, word2index, vocabulary_size, embedding_size):
    VOCABULARY_SIZE = vocabulary_size
    EMBEDDING_SIZE = embedding_size
    word2id = word2index
    embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))
    for word, index in tqdm(word2id.items()):
        try:
            embedding_weights[index, :] = word2vec[word]
        except KeyError:
            pass
    return embedding_weights
