import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.cm as cm

IMG_SIZE = 128
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'tumor_model.h5')

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def find_tumor_bounding_box(heatmap, original_img):
    # Threshold heatmap
    heatmap_uint8 = np.uint8(255 * heatmap)
    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap_uint8, (original_img.shape[1], original_img.shape[0]))
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(heatmap_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Draw bounding box on original image
        marked_img = original_img.copy()
        cv2.rectangle(marked_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return marked_img
    return original_img

loaded_model = None

def get_model():
    global loaded_model
    if loaded_model is None:
        if os.path.exists(MODEL_PATH):
            loaded_model = load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    return loaded_model

def predict_mri(image_path):
    model = get_model()
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Prepare image for model
    processed_img = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(processed_img, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    # Predict
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    confidence = preds[0][pred_index] * 100
    predicted_class = CLASSES[pred_index]
    
    result = {
        'class': predicted_class.title() if predicted_class != 'notumor' else 'No Tumor',
        'confidence': confidence,
        'original_img': orig_img,
        'marked_img': orig_img
    }
    
    if predicted_class != 'notumor':
        try:
            # Generate Grad-CAM Heatmap
            last_conv_layer = get_last_conv_layer_name(model)
            if last_conv_layer:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index)
                
                # We can return the grad-cam super-imposed image OR the bounding box image
                # Let's return bounding box image as primary marking
                marked_img = find_tumor_bounding_box(heatmap, orig_img)
                result['marked_img'] = marked_img
                
                # Also saving grad-cam visualization into result if GUI wants to use it
                superimposed = apply_gradcam(image_path, heatmap)
                result['gradcam_img'] = np.array(superimposed)
                
        except Exception as e:
            print(f"Warning: Failed to generate Grad-CAM marking: {e}")
            
    return result

if __name__ == "__main__":
    # For testing the script standalone
    print("Testing predictor...")
    test_img = os.path.join(BASE_DIR, 'Testing', 'glioma', 'Te-glTr_0000.jpg')
    if os.path.exists(test_img):
        print(f"Evaluating {test_img}")
        try:
            res = predict_mri(test_img)
            print(f"Result: {res['class']} ({res['confidence']:.2f}%)")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Note: Provide a test image to evaluate standalone.")
