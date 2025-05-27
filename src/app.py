from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Initialize FastAPI app
app = FastAPI(title="Leaf Disease Classifier")

# Load your model once on startup
model = load_model("registery\leaf_disease_color_model.h5")

# Define class labels
class_labels = [
    "Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple__Healthy",
    "Blueberry__Healthy", "Cherry(including_sour)Powdery_mildew", "Cherry(including_sour)_Healthy",
    "Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot", "Corn(maize)_Common_rust",
    "Corn_(maize)Northern_Leaf_Blight", "Corn(maize)Healthy", "Grape__Black_rot",
    "Grape__Esca(Black_Measles)", "Grape__Leaf_blight(Isariopsis_Leaf_Spot)", "Grape___Healthy",
    "Orange__Haunglongbing(Citrus_greening)", "Peach__Bacterial_spot", "Peach__Healthy",
    "Pepper,bell_Bacterial_spot", "Pepper,_bell_Healthy", "Potato__Early_blight",
    "Potato__Late_blight", "Potato_Healthy", "Raspberry_Healthy", "Soybean__Healthy",
    "Squash__Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry__Healthy",
    "Tomato__Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot", "Tomato__Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato___Healthy"
]

# API route to test the app
@app.get("/")
def read_root():
    return {"message": "Leaf Disease Detection API is running!"}


# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((128, 128))

        # Convert image to array
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        predicted_label = class_labels[predicted_index]

        return JSONResponse({
            "filename": file.filename,
            "predicted_label": predicted_label,
            "confidence": float(np.max(prediction[0]))
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))