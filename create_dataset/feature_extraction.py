# create_dataset/feature_extraction.py

import numpy as np

def extract_features(joint):
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

    features = np.concatenate([joint.flatten(), angle])
    return features
