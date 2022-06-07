Linformer_CONFIG = {
    'DATA_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/datasets/EyePacs',
    'SAVE_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/linformer_vit.pt',
    'RECORD_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/linformer_vit.rec',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 512,
    'PATCH_SIZE':32,
    'NUM_CLASSES':5,
    'BATCH_SIZE': 128,
    'EPOCHS': 1,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    },
    'Linformer':{
        'dim':128,
        'seq_len':257,
        'depth':12,
        'heads':8,
        'k':64,
        'patch_size':32,
        'num_classes':5,
    },
    "NUM_WORKERS": 12
}

Simple_CONFIG = {
    'DATA_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/datasets/EyePacs',
    'SAVE_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/simple_vit.pt',
    'RECORD_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/simple_vit.rec',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 512,
    'PATCH_SIZE':32,
    'NUM_CLASSES':5,
    'BATCH_SIZE': 128,
    'EPOCHS': 1,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    },
    'simple':{
        'dim':128,
        'image_size':512,
        'patch_size':32,
        'depth':12,
        'heads':8,
        'num_classes':5,
        'mlp_dim':1042
    },

    "NUM_WORKERS": 12
}

Vanilla_CONFIG = {
    'DATA_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/datasets/EyePacs',
    'SAVE_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/vanilla_vit.pt',
    'RECORD_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/vanilla_vit.rec',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 512,
    'PATCH_SIZE':32,
    'NUM_CLASSES':5,
    'BATCH_SIZE': 128,
    'EPOCHS': 1,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    },
    'vanilla':{
        'dim':128,
        'image_size':512,
        'patch_size':32,
        'dropout':0.1,
        'depth':12,
        'heads':8,
        'pool':'cls',
        'num_classes':5,
        'mlp_dim':1042
    },

    "NUM_WORKERS": 12
}

Conv_ViT_CONFIG = {
    'DATA_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/datasets/EyePacs',
    'SAVE_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/Conv_ViT.pt',
    'RECORD_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/Conv_ViT.rec',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 512,
    'PATCH_SIZE':32,
    'NUM_CLASSES':5,
    'BATCH_SIZE': 128,
    'EPOCHS': 1,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    },
    'CvT':{
        'num_classes':5,
        'dropout':0.1
    },

    "NUM_WORKERS": 12
}
Dist_ViT_CONFIG = {
    'DATA_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/datasets/EyePacs',
    'SAVE_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/distil_vit.pt',
    'RECORD_PATH': '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/distil_vit.rec',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 512,
    'PATCH_SIZE':32,
    'NUM_CLASSES':5,
    'BATCH_SIZE': 128,
    'EPOCHS': 1,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    },
    'distill':{
        'dim':128,
        'image_size':512,
        'patch_size':32,
        'depth':12,
        'heads':8,
        'num_classes':5,
        'mlp_dim':1042,
        'dropout':0.1,
    },

    "NUM_WORKERS": 12
}
