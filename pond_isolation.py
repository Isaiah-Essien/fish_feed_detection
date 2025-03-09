import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "fish_img_44.png"  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection to find edges
edges = cv2.Canny(blurred, 30, 100)

# Find contours
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour, assuming it's the pond
largest_contour = max(contours, key=cv2.contourArea)

# Get bounding box around the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the image to the bounding box
cropped_pond = image_rgb[y:y+h, x:x+w]

# Save the cropped output
output_cropped_path = "cropped_pond.png"
cv2.imwrite(output_cropped_path, cv2.cvtColor(cropped_pond, cv2.COLOR_RGB2BGR))

print(f"Cropped image saved as: {output_cropped_path}")
