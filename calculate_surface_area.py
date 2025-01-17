import numpy as np


def calculate_white_pixel_percentage(predicted_mask):
    """
    Calculate the percentage of white pixels in the predicted mask.

    Args:
        predicted_mask (numpy array): Predicted binary mask (0 to 1 or 0 to 255).

    Returns:
        float: Percentage of white pixels in the mask.
    """
    # Ensure the mask is binary (values 0 or 255)
    if predicted_mask.max() <= 1:  # If mask values are normalized (0 to 1)
        predicted_mask = (predicted_mask > 0.5).astype(
            np.uint8) * 255  # Threshold and scale
    else:  # If mask values are in range 0 to 255
        predicted_mask = (predicted_mask > 127).astype(
            np.uint8) * 255  # Threshold for binary mask

    # Count white pixels (value 255)
    white_pixels = np.sum(predicted_mask == 255)

    # Count total pixels
    total_pixels = predicted_mask.size

    # Calculate percentage of white pixels
    percentage_white = (white_pixels / total_pixels) * 100

    return percentage_white


# Example Usage
# Example binary mask (replace with your mask)
predicted_mask = np.random.randint(0, 2, (256, 256))
percentage_white = calculate_white_pixel_percentage(predicted_mask)

print(f"Percentage of white pixels (surface area): {percentage_white:.2f}%")
