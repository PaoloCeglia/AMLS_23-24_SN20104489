import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, callbacks

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve


import matplotlib.pyplot as plt
import seaborn as sn



class BuildCNN:
    """Class for building the CNN model"""
    @staticmethod
    def build_model():
        model = models.Sequential([
            
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), #Convolutional layer
            layers.MaxPooling2D((2, 2)), #Max pooling layer
            
            layers.Flatten(),   #Flatenning layer
            layers.Dense(64, activation='relu'),  #Dense layer
            layers.Dense(1, activation='sigmoid') #Output layer
        ])
        return model

class CNNModel:
    """Preprocessing of the data, training, evaluating and testing the model"""

    def __init__(self, file_path):
        self.load_data(file_path)
        self.model = BuildCNN.build_model()

    def load_data(self, path):
        data = np.load(path)
        self.train_images, self.val_images, self.test_images = data['train_images'], data['val_images'], data['test_images']
        self.train_labels, self.val_labels, self.test_labels = data['train_labels'], data['val_labels'], data['test_labels']

    def preprocessingData(self):
        self.train_images = self.dataPrep(self.train_images)
        self.val_images = self.dataPrep(self.val_images)
        self.test_images = self.dataPrep(self.test_images)
    
    @staticmethod
    def dataPrep(data):                        #Normalizes and expands the dimensions of the data
        norm_data = data.astype('float32') / 255  #Normalizes
        return np.expand_dims(norm_data, axis=-1) #Expands Data
    


    def data_augmentation(self):       #Data augmentation for better training and generalization
        self.augmented_data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
            )



    def train_model(self):

        self.data_augmentation() 

        patience = 6
        epochs = 40
        batch_size= 32 

        #Compiles the model and introduces early stopping
        self.model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights=True)
        
        
        history = self.model.fit(
            self.augmented_data_gen.flow(self.train_images, self.train_labels, batch_size),
            validation_data=(self.val_images, self.val_labels),
            epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1
        )
        self.model.save('pretrained_CNNA.h5')   #Remove comment to save a new model
        self.plot_training_history(history)
        

    def evaluate_and_display_metrics(self):
        predictions = self.model.predict(self.test_images)

        # Computes accuracy
        accuracy = accuracy_score(self.test_labels, predictions.round())
        print(f'Test Accuracy: {accuracy:.4f}')

        # Displays classification report
        print('Classification Report:')
        print(classification_report(self.test_labels, predictions.round()))



        
        precision, recall, _ = precision_recall_curve(self.test_labels, predictions)

        # Plotting a precision recall curve

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()



        # Calculates and shows the confusion matrix
        cmatrix = confusion_matrix(self.test_labels, predictions.round())
        plt.figure(figsize=(6, 6))
        sn.heatmap(cmatrix, annot=True, fmt='d', cmap='Paired', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Calculates and shows ROC curve and AUC
        fpRate, tpRate, _ = roc_curve(self.test_labels, predictions)
        auc_value = roc_auc_score(self.test_labels, predictions)

        plt.figure(figsize=(8, 6))
        plt.plot(fpRate, tpRate, label=f'ROC Curve (AUC = {auc_value:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
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
        plt.tight_layout()
        plt.show()

# Functions to be called
def trainCNNA(data_path):
    cnn_model = CNNModel(data_path)
    cnn_model.preprocessingData()
    cnn_model.train_model()
    cnn_model.evaluate_and_display_metrics()

def test_trained_model_A(model_path, data_path):
    cnn_model = CNNModel(data_path)
    cnn_model.preprocessingData()
    cnn_model.model = tf.keras.models.load_model(model_path)
    cnn_model.evaluate_and_display_metrics()
