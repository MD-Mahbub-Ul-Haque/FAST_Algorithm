import cv2
import numpy as np

scl = 50  # Scaling factor for display
image_path = r"C:\Users\User\Desktop\openCv\rover03.jpg"

# Load the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the path!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny Edge Detection to identify rough terrain
edges = cv2.Canny(blurred, 50, 150)

# Find contours (edges of rough terrain)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an overlay for markings
overlay = image.copy()

# Create a mask for smooth areas
smooth_mask = np.ones_like(gray) * 255  # Start with a white image

# Fill rough areas (detected by contours) with black
cv2.drawContours(smooth_mask, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

# Apply the mask to highlight smooth regions
smooth_area = cv2.bitwise_and(image, image, mask=smooth_mask)

# Bounding box grouping for merging overlapping areas
merged_contours = []
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Flag to check if this contour should be merged
    merged = False
    for j, merged_cnt in enumerate(merged_contours):
        mx, my, mw, mh = cv2.boundingRect(merged_cnt)
        
        # Check if the bounding boxes overlap or are close to each other
        if (x < mx + mw and x + w > mx and y < my + mh and y + h > my):
            # Merge the contours by combining the points
            merged_contours[j] = np.concatenate([merged_cnt, cnt])
            merged = True
            break
    
    # If no overlap, create a new merged contour
    if not merged:
        merged_contours.append(cnt)

# Create an overlay with merged contours
for cnt in merged_contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)  # Get bounding box

    if area < 1000:
        label = "Sandy Surface"
        color = (200, 200, 100)  # Light brown
    elif area < 5000:
        label = "Plain Surface"
        color = (0, 255, 0)  # Green
    elif area < 15000:
        label = "Rocky Surface"
        color = (255, 140, 0)  # Orange
    else:
        label = "Very Rough Surface"
        color = (255, 0, 0)  # Red
    
    # Fill the merged terrain area with transparent color
    cv2.drawContours(overlay, [cnt], -1, color, thickness=cv2.FILLED)

    # Draw a frame (border) around the merged terrain
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)

    # Label the merged terrain inside the bounding box
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# Apply transparency (70% transparent overlay)
alpha = 0.3
cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

# Resize the output image
width = int(image.shape[1] * scl / 100)
height = int(image.shape[0] * scl / 100)
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Display the processed image
cv2.imshow("Rover Terrain Classification with Merged Areas", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
