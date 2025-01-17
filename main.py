from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import cv2
import base64
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("./best_model.keras")


def preprocess_image(image_bytes, target_size=(256, 256)):
    """
    Preprocess the uploaded image for prediction.
    Args:
        image_bytes: Bytes of the uploaded image.
        target_size: Desired size for the image (height, width).

    Returns:
        numpy array: Preprocessed image ready for the model.
    """
    image = cv2.imdecode(np.frombuffer(
        image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file")
    image = cv2.resize(image, target_size)
    image = image / 255.0 
    return np.expand_dims(image, axis=0)


def calculate_white_pixel_percentage(predicted_mask):
    """
    Calculate the percentage of white pixels in the predicted mask.
    Args:
        predicted_mask (numpy array): Predicted binary mask (0 or 255).

    Returns:
        float: Percentage of white pixels in the mask.
    """
    binary_mask = (predicted_mask > 127).astype(np.uint8) * 255
    white_pixels = np.sum(binary_mask == 255)
    total_pixels = binary_mask.size
    percentage_white = (white_pixels / total_pixels) * 100

    return percentage_white


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the mask and calculate white pixel percentage.
    Args:
        file: Uploaded image file.

    Returns:
        JSON: Percentage of white pixels and the Base64-encoded mask.
    """
    try:
        contents = await file.read()
        preprocessed_image = preprocess_image(contents)
        predicted_mask = model.predict(preprocessed_image)[
            0]

        predicted_mask = (predicted_mask * 255).astype(np.uint8)
        percentage_white = calculate_white_pixel_percentage(predicted_mask)

        _, buffer = cv2.imencode(".png", predicted_mask)
        mask_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "Area covered with feed": percentage_white,
            "predicted_mask": mask_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
