import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras import layers, models, callbacks
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn

class BuildCNN:
    """Class for building the CNN model"""
    @staticmethod
    def build_model():
        model = models.Sequential([
            
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)), #convolution layer 1
            layers.MaxPooling2D((2, 2)), #Maxpoolinglayer 1
            layers.Conv2D(64, (3, 3), activation='relu'), #Convolution layer 2
            layers.MaxPooling2D((2, 2)), #Maxpoolinglayer 2
            layers.Flatten(), #Flatten layer
            layers.Dense(128, activation='relu'), #Dense layer
            layers.Dense(9, activation='softmax')  #Output layer with softmax for multi class
        ])
        return model

class CNNModel:
    """Class for handling data loading, preprocessing, training, and evaluating the model"""
    def __init__(self, file_path):
        self.load_data(file_path)
        self.model = BuildCNN.build_model()  

    def load_data(self, path):  #Loads data
        data = np.load(path)
        self.train_images, self.val_images, self.test_images = data['train_images'], data['val_images'], data['test_images']
        self.train_labels, self.val_labels, self.test_labels = data['train_labels'], data['val_labels'], data['test_labels']

    def preprocessingData(self):   #Preprocessing
        self.train_images = self.dataPrep(self.train_images)
        self.val_images = self.dataPrep(self.val_images)
        self.test_images = self.dataPrep(self.test_images)

    @staticmethod
    def dataPrep(data):
        norm_data = data.astype('float32') / 255  # Normalizes the data
        return norm_data  
    
    
    def train_model(self):
        
        epochs = 15
        batch_size = 256
        patience = 4

        #Compiles the model and introduces early stopping
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights=True)
        
    
        history = self.model.fit(
            self.train_images, self.train_labels,
            validation_data=(self.val_images, self.val_labels),
            epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1
        )
        #self.model.save('pretrained_CNNB.h5')   #Remove comment to save a new mode
        self.plot_training_history(history)


    def evaluate_and_display_metrics(self): #Function to calculate and display the metrics

        predictions = self.model.predict(self.test_images)
        predicted_labels = np.argmax(predictions, axis=1)

        # Computes the accuracy
        accuracy = accuracy_score(self.test_labels, predicted_labels)
        print(f'Test Accuracy: {accuracy:.4f}')

        # Displays the classification report
        print('Classification Report:')
        print(classification_report(self.test_labels, predicted_labels))

        # Calculates and shows confusion matrix
        cmatrix = confusion_matrix(self.test_labels, predicted_labels)
        plt.figure(figsize=(10, 10))
        sn.heatmap(cmatrix, annot=True, fmt='d', cmap='rocket', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    @staticmethod
    def plot_training_history(history): 

        plt.figure(figsize=(12, 5))
        
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over epochs')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

""" Functions to be called by main """
def trainCNNB(data_path):
    cnn_model = CNNModel(data_path)
    cnn_model.preprocessingData()
    cnn_model.train_model()
    cnn_model.evaluate_and_display_metrics()

def test_trained_model_B(model_path, data_path):
    cnn_model = CNNModel(data_path)
    cnn_model.preprocessingData()
    cnn_model.model = tf.keras.models.load_model(model_path)
    cnn_model.evaluate_and_display_metrics()
