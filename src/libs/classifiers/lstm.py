
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras import Model
from libs.data_processor import DATA_PROCESSOR
from .base import BaseClassifier
from ..utils import evaluate
import numpy as np
    
class LSTMClassifer(BaseClassifier):
    def __init__(self):
        self.model = None
        self.method_name = "lstm"
        
    def build_model(self, **kwargs):
        """
        Builds and compiles a Bidirectional LSTM model for text classification.
        Keyword Arguments:
        max_len (int): Maximum length of input sequences. Default is 100.
        vocab_size (int): Size of the vocabulary. Default is 10000.
        embedding_size (int): Dimension of the embedding vectors. Default is 64.
        The model architecture includes:
        - An Embedding layer
        - A Conv1D layer with ReLU activation
        - A MaxPooling1D layer
        - A Bidirectional LSTM layer
        - A Dropout layer
        - A Dense output layer with softmax activation
        The model is compiled with the Adam optimizer, categorical crossentropy loss, and accuracy metric.
        """
        
        max_len = 128
        vocab_size = 10000
        embedding_size = 64
        if kwargs.get('max_len'):
            max_len = kwargs.get('max_len')
        if kwargs.get('vocab_size'):
            vocab_size = kwargs.get('vocab_size')
        if kwargs.get('embedding_size'):
            embedding_size = kwargs.get('embedding_size')
        
        """
        The model architecture includes:
        - An Embedding layer
        - A Conv1D layer with ReLU activation
        - A MaxPooling1D layer
        - A Bidirectional LSTM layer
        - A Dropout layer
        - A Dense output layer with softmax activation
        """
        inputs = Input(shape=(max_len,))
        embedding_layer = Embedding(vocab_size, embedding_size, input_length=max_len)(inputs)
        conv1d_layer = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedding_layer)
        maxpooling_layer = MaxPooling1D(pool_size=2)(conv1d_layer)
        bidirectional_lstm_layer = Bidirectional(LSTM(32))(maxpooling_layer)
        dropout_layer = Dropout(0.4)(bidirectional_lstm_layer)
        outputs = Dense(3, activation='softmax')(dropout_layer)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    def print(self):
        print(self.model.summary())
        
    def train(self, data_processor: DATA_PROCESSOR, **kwargs):
        
        batch_size = kwargs.get('batch_size') if kwargs.get('batch_size') else 32
        epochs = kwargs.get("epochs") if kwargs.get("epochs") else 100

        early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

        #print(np.shape(data_processor.X_train))
        #print(data_processor.Y_eval)
        #dsadaddsada
        self.model.fit(data_processor.X_train, data_processor.Y_train,
                            validation_data=(data_processor.X_eval, data_processor.Y_eval),
                            batch_size=batch_size, epochs=epochs, verbose=1,
                            shuffle=True,
                            callbacks=[early_stop])
        
    def evaluate(self, data_processor: DATA_PROCESSOR):
        y_pred = self.model.predict(data_processor.X_test)
        evaluate (y_true=np.argmax(data_processor.Y_test, axis=1), y_pred=np.argmax(y_pred, axis=1), method= self.method_name)