# Fish Feed Detection API

This project provides a FastAPI-based web service for analyzing fish pond images to detect the area covered with feed particles. The service uses a trained deep learning model to generate segmentation masks and calculate the percentage of feed-covered areas relative to the total pond surface.

---

## Features
- **Mask Prediction**: Generates a segmentation mask from uploaded images using a pre-trained TensorFlow `.keras` model.
- **White Pixel Percentage Calculation**: Calculates the percentage of white pixels (representing feed particles) in the generated mask.
- **Mask Visualization**: Returns the predicted mask in Base64 format, allowing for direct visualization or debugging.
- **FastAPI Framework**: A lightweight and high-performance web API for easy integration.

---

## How It Works
1. **Input**: Upload an image of the fish pond.
2. **Processing**:
   - The image is preprocessed (resized, normalized) to match the model's input requirements.
   - The model predicts a binary segmentation mask where white pixels represent feed particles.
   - The percentage of feed-covered areas is calculated based on the white pixels in the mask.
3. **Output**:
   - The percentage of the area covered with feed.
   - The predicted mask, encoded in Base64 format for visualization.

---


## Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.13.0 (or compatible version)
- FastAPI and Uvicorn

### Steps
1. Clone the repository:
   ```bash
   cd fish-feed-detection

- run `pip install -r requirements.txt`
- run `uvicorn main:app --reload`


### fish-feed-detection/
├── main.py                # Main FastAPI application
├── requirements.txt       # Python dependencies
├── path_to_your_model.keras # Trained segmentation model
└── README.md              # Project documentation
