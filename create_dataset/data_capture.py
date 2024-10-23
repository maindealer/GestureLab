# create_dataset/data_capture.py

import cv2
import mediapipe as mp
import numpy as np
import time
import os

from feature_extraction import extract_features # Import the feature extraction function


def capture_data(action_name, collection_time):
    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)

    # Format the current time as YYMMDDHHMMSS
    created_time = time.strftime("%y%m%d%H%M%S")
    dataset_dir = os.path.join('create_dataset', 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    data = []

    # Display countdown before data collection
    for i in range(3, 0, -1):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        # Display message in the center
        message1 = 'Prepare your gesture. Data collection will start soon.'
        message2 = f'Remaining time: {i} seconds'
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate text sizes
        text_size1, _ = cv2.getTextSize(message1, font, 0.8, 2)
        text_size2, _ = cv2.getTextSize(message2, font, 0.8, 2)

        # Calculate text positions
        text_x1 = int((img.shape[1] - text_size1[0]) / 2)
        text_y1 = int((img.shape[0] + text_size1[1]) / 2) - 30
        text_x2 = int((img.shape[1] - text_size2[0]) / 2)
        text_y2 = int((img.shape[0] + text_size2[1]) / 2) + 30

        cv2.putText(img, message1, (text_x1, text_y1), font, 0.8, (255, 255, 255), 2)
        cv2.putText(img, message2, (text_x2, text_y2), font, 0.8, (255, 255, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

    start_time = time.time()

    while time.time() - start_time < collection_time:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Display remaining time
        elapsed_time = time.time() - start_time
        remaining_time = int(collection_time - elapsed_time) + 1
        cv2.putText(img, f'Remaining time: {remaining_time} seconds', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Use the extract_features function
                d = extract_features(joint)

                data.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    data = np.array(data)
    print(f"Data collection for action '{action_name}' completed: {data.shape}")
    np.save(os.path.join(dataset_dir, f'raw_{action_name}_{created_time}.npy'), data)

    # Create sequence data
    seq_length = 30
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(f"Sequence data shape for action '{action_name}': {full_seq_data.shape}")
    np.save(os.path.join(dataset_dir, f'seq_{action_name}_{created_time}.npy'), full_seq_data)