from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('/content/drive/MyDrive/Pneumonia/trained_model.keras')

def load_and_preprocess_image(image_path):
    try:
        # Load image from the specified path
        img = image.load_img(image_path, target_size=(120, 120))
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None

    # Convert image to NumPy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to be between 0 and 1
    img_array /= 255.0
    
    return img_array

def predict_pneumonia(image_path):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)

    # Check if image loading was successful
    if img_array is None:
        return

    # Make prediction
    prediction = model.predict(img_array)

    # Display the image
    img = image.load_img(image_path, target_size=(120, 120))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Display the prediction result
    if prediction[0, 0] >= 0.5:
        result = "Pneumonia"
    else:
        result = "No Pneumonia"

    accuracy = prediction[0, 0] if result == "Pneumonia" else 1 - prediction[0, 0]

    print(f"Prediction: {result}")
    print(f"Accuracy: {accuracy:.2%}")

# Example usage with a local image path
image_path = "img.jpeg"
predict_pneumonia("/content/drive/MyDrive/Pneumonia/img.jpeg")
