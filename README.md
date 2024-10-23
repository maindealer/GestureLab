# GestureLab
동적 제스처 인식 

# 프로젝트 구조
├── create_dataset/
│   ├── main.py              # 데이터 수집을 위한 GUI 스크립트
│   ├── data_capture.py      # 데이터 수집 모듈
│   └── dataset/             # 수집된 데이터가 저장되는 디렉토리
│       ├── raw_*.npy        # 원본 데이터 파일들
│       └── seq_*.npy        # 시퀀스 데이터 파일들
├── train.ipynb              # 모델 학습을 위한 Jupyter Notebook
├── test.py                  # 학습된 모델을 사용하여 실시간 예측하는 스크립트
├── models/                  # 학습된 모델이 저장되는 디렉토리
│   └── best_model.keras     # 가장 성능이 좋은 모델 파일
└── actions.npy              # 액션 이름들을 저장한 파일