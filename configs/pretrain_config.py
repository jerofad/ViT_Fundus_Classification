CNN_CONFIG = {
    'MODEL_NAME': '',
    'DATA_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/datasets/EyePacs',
    'SAVE_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/models/cnn/',
    'RECORD_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/records/cnn/',
    'LEARNING_RATE': 1e-3,
    'INPUT_SIZE': 224,
    'BATCH_SIZE': 64,
    'EPOCHS': 1,
    'FEATURE_DIM': 2048,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    },
    "NUM_WORKERS": 12

}

VIT_CONFIG = {
    'MODEL_NAME': '',
    'DATA_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/datasets/EyePacs',
    'SAVE_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/models/ViT/',
    'RECORD_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/records/ViT/',
    'LEARNING_RATE': 1e-3,
    'INPUT_SIZE': 224,
    'BATCH_SIZE': 128,
    'EPOCHS': 1,
    'FEATURE_DIM': 2048,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    },
    "NUM_WORKERS": 12

}