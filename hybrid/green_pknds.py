'''This script will substitute the model '''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========== Load Image ===========
image_path = "/Users/ntohsi/development/original_images/fish_img177.jpg"  # Change this to your image file path
image = cv2.imread(image_path)


if image is None:
    raise ValueError("Error: Image not found. Please check the file path.")

# Resize image for faster processing (optional)
scale_percent = 70  # Resize to 70% of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image = cv2.resize(image, (width, height))

# =========== Step 1: Apply GrabCut for Background Removal ===========
mask = np.zeros(image.shape[:2], np.uint8)

# Define a rough rectangle around the pond (manually adjustable)
rect = (20, 50, width - 30, height - 40)

# Background and foreground models for GrabCut
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Apply GrabCut
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Convert mask to binary (1 = foreground, 0 = background)
mask_bin = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

# Apply mask to extract only the pond
pond_segmented = image * mask_bin[:, :, np.newaxis]

# =========== Step 2: Convert to HSV & Apply Floating Feed Color Segmentation ===========
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Fine-tuned HSV range for floating feed (adjust as needed)
lower_feed = np.array([30, 50, 107])  # Lower bound
upper_feed = np.array([35, 255, 255])  # Upper bound

# Create mask for floating feed
mask_feed = cv2.inRange(hsv_image, lower_feed, upper_feed)

# Apply segmentation mask to remove background interference
mask_feed = mask_feed * mask_bin

# =========== Step 3: Apply Morphological Operations ===========
kernel = np.ones((3, 3), np.uint8)
mask_cleaned = cv2.morphologyEx(mask_feed, cv2.MORPH_OPEN, kernel)

# =========== Step 4: Remove Large Non-Feed Objects ===========
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)

# Define size limits for feed particles
min_size = 10  # Minimum size of feed particles
max_size = 500  # Ignore large objects (reflections, pipes, etc.)

feed_mask_final = np.zeros_like(mask_cleaned)

for i in range(1, num_labels):  # Ignore background (label 0)
    area = stats[i, cv2.CC_STAT_AREA]
    if min_size <= area <= max_size:  # Keep only valid floating feed
        feed_mask_final[labels == i] = 255

# =========== Step 5: Use Blob Detection for Improved Precision ===========
# Configure SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 5  # Minimum feed size
params.maxArea = 300  # Ignore large objects

params.filterByCircularity = True
params.minCircularity = 0.4  # Floating feed should be fairly circular

params.filterByInertia = True
params.minInertiaRatio = 0.2

# Create a blob detector
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the feed mask
keypoints = detector.detect(feed_mask_final)

# Draw detected blobs as circles
blob_image = pond_segmented.copy()
for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    cv2.circle(blob_image, (x, y), int(kp.size / 2), (0, 255, 0), 2)

# =========== Step 6: Compute Accurate Feed Coverage ===========
# Count number of feed pixels in `feed_mask_final`
feed_pixels = np.count_nonzero(feed_mask_final)

# Count total pond area (foreground pixels from GrabCut)
total_pond_pixels = np.count_nonzero(mask_bin)

# Calculate feed coverage percentage
feed_coverage = (feed_pixels / total_pond_pixels) * 100

print(f"Floating Feed Coverage: {feed_coverage:.2f}%")

# =========== Step 7: Display Results ===========
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Pond Image")
axs[0].axis("off")

axs[1].imshow(cv2.cvtColor(pond_segmented, cv2.COLOR_BGR2RGB))
axs[1].set_title("Segmented Pond (GrabCut)")
axs[1].axis("off")

axs[2].imshow(feed_mask_final, cmap="gray")
axs[2].set_title("Filtered Floating Feed")
axs[2].axis("off")

axs[3].imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))
axs[3].set_title("Feed Detection Overlay")
axs[3].axis("off")

plt.show()