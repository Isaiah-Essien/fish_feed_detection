from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tempfile
import os
import webbrowser

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
model = load_model("./best_model.keras")


def preprocess_image(image_bytes, target_size=(256, 256)):
    """
    Preprocess the uploaded image for prediction.
    Args:
        image_bytes: Bytes of the uploaded image.
        target_size: Desired size for the image (height, width).

    Returns:
        tuple: Preprocessed image ready for the model and the original image for visualization.
    """
    image = cv2.imdecode(np.frombuffer(
        image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file")
    # Convert for visualization
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0), original_image


def predict_image(image_bytes, model, target_size=(256, 256), threshold=0.5):
    """
    Predict the segmentation mask for a given image.
    Args:
        image_bytes: Bytes of the input image.
        model: Trained segmentation model.
        target_size (tuple): Size to resize the input image.
        threshold (float): Threshold for binarizing the predicted mask.

    Returns:
        tuple: Predicted binary mask and the original image for visualization.
    """
    preprocessed_image, original_image = preprocess_image(
        image_bytes, target_size)
    prediction = model.predict(preprocessed_image)[0]  # Remove batch dimension
    binary_mask = (prediction > threshold).astype(
        np.uint8)  # Binarize the mask
    return binary_mask, original_image


def visualize_prediction(original_image, predicted_mask):
    """
    Visualize the input image and predicted segmentation mask.
    Args:
        original_image (numpy array): Original input image.
        predicted_mask (numpy array): Predicted binary mask.

    Returns:
        str: Path to the saved visualization image.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap="gray")
    plt.axis("off")

    temp_file_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".png").name
    plt.savefig(temp_file_path)
    plt.close()
    return temp_file_path


@app.post("/visualize_mask_and_img/")
async def visualize_mask_and_img(file: UploadFile = File(...)):
    """
    Endpoint to visualize the input image and predicted segmentation mask.
    Args:
        file: Uploaded image file.

    Returns:
        JSON: Message indicating the visualization was opened in a browser.
    """
    try:
        contents = await file.read()
        predicted_mask, original_image = predict_image(contents, model)
        visualization_path = visualize_prediction(
            original_image, predicted_mask)

        # Open the visualization in the default web browser
        webbrowser.open("file://" + visualization_path)

        return {"message": "Visualization opened in a new browser tab."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the mask and calculate white pixel percentage.
    Args:
        file: Uploaded image file.

    Returns:
        JSON: Percentage of white pixels in the mask.
    """
    try:
        contents = await file.read()
        preprocessed_image = cv2.imdecode(
            np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if preprocessed_image is None:
            raise ValueError("Invalid image file")
        preprocessed_image = cv2.resize(preprocessed_image, (256, 256))
        preprocessed_image = preprocessed_image / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        predicted_mask = model.predict(preprocessed_image)[0]
        predicted_mask = (predicted_mask * 255).astype(np.uint8)

        binary_mask = (predicted_mask > 127).astype(np.uint8) * 255
        white_pixels = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size
        percentage_white = round((white_pixels / total_pixels) * 100,2)
        percentage_white=percentage_white*0.9
        
        return {"Area_with_feed": percentage_white}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
