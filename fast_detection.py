import cv2
import numpy as np
import mediapipe as mp

def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip image for better tracking
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark points
            points = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append((x, y))
            
            # Convex hull for gesture detection
            if len(points) > 0:
                hull = cv2.convexHull(np.array(points))
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                
                # Detect gesture based on defects
                hull_idx = cv2.convexHull(np.array(points), returnPoints=False)
                defects = cv2.convexityDefects(np.array(points), hull_idx)
                
                if defects is not None:
                    count_defects = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start, end, far = points[s], points[e], points[f]
                        if d > 10000:
                            count_defects += 1
                            cv2.circle(frame, far, 5, (0, 0, 255), -1)
                    
                    # Hand gesture classification
                    if count_defects == 0:
                        text = "Fist"
                    elif count_defects == 1:
                        text = "Pointing"
                    else:
                        text = "Open Hand"
                    
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Detect edges for better visualization
    edges = detect_edges(frame)
    
    # Combine edge detection with original frame
    combined = cv2.addWeighted(frame, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)
    
    cv2.imshow("Hand Tracking and Edge Detection", combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
