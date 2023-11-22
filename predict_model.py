from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import sys

# Load the model
model = load_model('model/leaf_classifier_model_agumented_65.h5')


# predefined classes
classes = ['Burlant', 'Buttnera', 'Kordia', 'Rivan', 'Sam', 'Summit', 'Van', 'Vega']


# Load and preprocess the image
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Check if the image file path is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python predict_leaf.py <image_file>")
    sys.exit(1)

# Get the image file path from the command line
image_path = sys.argv[1]

# Specify the target size based on your model's input size
target_size = (224, 224)

# Preprocess the image and make predictions
input_image = preprocess_image(image_path, target_size)
predictions = model.predict(input_image)

# Assuming 'classes' is your list of class labels
predicted_class_index = np.argmax(predictions)
predicted_class = classes[predicted_class_index]

print(f"Predicted class: {predicted_class}")