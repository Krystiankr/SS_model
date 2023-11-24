from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = FastAPI()

# Load the model
model = load_model('model/leaf_classifier_model_agumented_65.h5')

# predefined classes
classes = ['Burlant', 'Buttnera', 'Kordia', 'Rivan', 'Sam', 'Summit', 'Van', 'Vega']

def preprocess_image(image_file, target_size):
    img = load_img(image_file, target_size=target_size)
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load the image
    image = await file.read()
    
    # Save image to a temporary file (if needed)
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(image)

    # Preprocess the image and make predictions
    target_size = (224, 224)
    input_image = preprocess_image(temp_file_path, target_size)
    predictions = model.predict(input_image)

    # Get predicted class
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]

    return {"Predicted class": predicted_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
