import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Constants
IMG_SIZE = 128
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'Training')
TEST_DIR = os.path.join(BASE_DIR, 'Testing')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data(directory):
    X = []
    y = []
    for label_idx, label_name in enumerate(CLASSES):
        folder_path = os.path.join(directory, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist.")
            continue
            
        for img_name in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # OpenCV reads as BGR, convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize to 128x128
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label_idx)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                
    # Normalize pixel values to [0, 1]
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y, dtype='int')
    return X, y

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'accuracy_loss.png'))
    plt.close()
    print("Saved accuracy/loss graph to accuracy_loss.png")

def plot_confusion_matrix(y_true, y_pred_classes):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
    plt.close()
    print("Saved confusion matrix to confusion_matrix.png")

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5), # Regularization
        Dense(len(CLASSES), activation='softmax') # Softmax for multi-class
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    print("Loading training data...")
    X_train, y_train = load_data(TRAIN_DIR)
    
    print("Loading testing data...")
    X_test, y_test = load_data(TEST_DIR)
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    print("Building model...")
    model = build_model()
    model.summary()
    
    model_save_path = os.path.join(MODEL_DIR, 'tumor_model.h5')
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    
    print("Training the model...")
    history = model.fit(
        X_train, y_train,
        epochs=20, # Run for all 20 epochs
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint] # Removed early_stop
    )
    
    print("Plotting training history...")
    plot_history(history)
    
    print("Evaluating model...")
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred_classes, target_names=CLASSES)
    print(report)
    
    with open(os.path.join(BASE_DIR, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
    
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred_classes)
    
    # If using newer tensorflow, save as .keras is recommended, but we stick to .h5 for clarity as requested
    model.save(model_save_path)
    print(f"Model successfully saved to {model_save_path}")
