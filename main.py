import cv2
import mediapipe as mp

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Define finger tip landmarks
FINGER_TIPS = [4, 8, 12, 16, 20]

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for better mirroring experience
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with Mediapipe
    results = hands.process(rgb_frame)

    total_fingers = 0
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get hand type (Left or Right)
            hand_type = results.multi_handedness[i].classification[0].label

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark

            # Count fingers for each hand
            fingers = []
            for tip in FINGER_TIPS:
                if tip == 4:  # Thumb case
                    if hand_type == "Right":  # Check normal way for right hand
                        fingers.append(1 if landmarks[tip].x < landmarks[tip - 1].x else 0)
                    else:  # Invert check for left hand
                        fingers.append(1 if landmarks[tip].x > landmarks[tip - 1].x else 0)
                else:
                    fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)

            total_fingers += fingers.count(1)  # Add finger count for each hand

    # Display total finger count
    cv2.putText(frame, f'Fingers: {total_fingers}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("HandTrackAI - Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
