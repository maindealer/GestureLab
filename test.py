# test.py

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# 액션 이름 불러오기
actions = np.load('actions.npy')
print(f'Loaded actions: {actions}')

seq_length = 30

# 학습된 모델 로드
model = load_model('models/final_model.keras')

# MediaPipe hands 모델 설정
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

            # # 관절 간 벡터 계산
            # v1 = joint[
            #     [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3
            # ]  # 부모 관절
            # v2 = joint[
            #     [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3
            # ]  # 자식 관절
            # v = v2 - v1  # [20, 3]
            # # 벡터 정규화
            # v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # # 벡터 간 각도 계산
            # angle = np.arccos(np.einsum(
            #     'nt,nt->n',
            #     v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
            #     v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
            # ))  # [15,]

            # angle = np.degrees(angle)  # 라디안에서 도로 변환

            # Updated code from data_capture.py
            # Compute angles between joints
            v1 = joint[
                [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3
            ]
            v2 = joint[
                [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3
            ]
            v = v2 - v1  # [20, 3]
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angles using arccos of dot product
            angle = np.arccos(np.einsum(
                'nt,nt->n',
                v[:-1, :],
                v[1:, :]
            ))
            angle = np.degrees(angle)  # Convert radian to degree
            ##################################################################

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

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

            # 마지막 3개의 액션이 동일한지 확인
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
            else:
                this_action = ''

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]),
                        int(res.landmark[0].y * img.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)

    else:
        action_seq.clear()
        last_action = None

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
