# test.py

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from feature_extraction import extract_features  # Import the feature extraction function

# Load action names
actions = np.load('actions.npy')
print(f'Loaded actions: {actions}')

seq_length = 30

# Load the trained model
model = load_model('models/final_model.keras')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img0 = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Use the extract_features function
            d = extract_features(joint)

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            print(f'Input data shape: {input_data.shape}')  # Verify data shape

            try:
                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]

                if last_action != action:
                    last_action = action
                    action_seq.clear()

                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                # Check if the last 3 actions are the same
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action
                else:
                    this_action = ''

                cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]),
                            int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2)
            except Exception as e:
                print(f'Error during prediction: {e}')
                continue

    else:
        action_seq.clear()
        last_action = None

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
