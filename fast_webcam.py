import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Initialize FAST Feature Detector
fast = cv2.FastFeatureDetector_create()

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints using FAST
    keypoints = fast.detect(gray, None)

    # Draw keypoints on the frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

    # Display the output
    cv2.imshow("FAST Corner Detection", frame_with_keypoints)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
