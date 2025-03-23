import cv2
import numpy as np

scl = 1000
n = 5 

# Load the image
image_path = r"C:\Users\User\Desktop\openCv\black-dot.jpg"  # Change this to your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the path!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to resize image for display
def resize_image(img, scale_percent=scl):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Step 1: Show the original grayscale image
cv2.imshow("Step 1: Grayscale Image", resize_image(gray))
cv2.waitKey(1000)

# Initialize FAST Feature Detector
fast = cv2.FastFeatureDetector_create()

# Detect keypoints
keypoints = fast.detect(gray, None)

# Step 2: Show the 16-pixel mask around selected keypoint (transparent overlay)
mask_image = image.copy()
overlay = mask_image.copy()

# Select the first keypoint
if keypoints:
    kp = keypoints[0]  # Select the first keypoint
    x, y = int(kp.pt[0]), int(kp.pt[1])

    # FAST 16-pixel circle for corner detection
    ring_offsets = [(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
                    (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)]
    
    for dx, dy in ring_offsets:
        cv2.circle(overlay, (x + dx, y + dy), 2, (0, 0, 255), -1)  # Red pixels in ring
    
    # Blend overlay for transparency
    alpha = 0.5
    mask_image = cv2.addWeighted(overlay, alpha, mask_image, 1 - alpha, 0)

    # Draw keypoint
    cv2.circle(mask_image, (x, y), 3, (0, 255, 0), -1)  # Green center point

cv2.imshow("Step 2: FAST Pixel Mask (Transparent)", resize_image(mask_image))
cv2.waitKey(1000)

# Step 3: Show selected pixels and intensity thresholding with transparency
comparison_image = image.copy()
overlay = comparison_image.copy()

if keypoints:
    intensity = gray[y, x]

    # Draw keypoint
    cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)

    # Draw intensity comparison pixels
    for dx, dy in ring_offsets:
        if 0 <= y + dy < gray.shape[0] and 0 <= x + dx < gray.shape[1]:
            px_intensity = gray[y + dy, x + dx]
            color = (255, 0, 0) if px_intensity > intensity + 10 else (0, 255, 255)  # Blue = brighter, Yellow = darker
            cv2.circle(overlay, (x + dx, y + dy), 2, color, -1)

    # Blend overlay for transparency
    comparison_image = cv2.addWeighted(overlay, alpha, comparison_image, 1 - alpha, 0)

cv2.imshow("Step 3: Intensity Comparisons (Transparent)", resize_image(comparison_image))
cv2.waitKey(1000)

# Step 4: Show final detected corner (Only One Keypoint from Step 2 and 3)
final_image = image.copy()

if keypoints:
    cv2.drawKeypoints(final_image, [keypoints[0]], final_image, color=(0, 255, 0))

cv2.imshow("Step 4: FAST Detected Corner (Single Keypoint)", resize_image(final_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
