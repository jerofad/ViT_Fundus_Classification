import torch
import torch.nn as nn

from torchvision.models import resnet50

# vision transformer models
from vit_pytorch.efficient import ViT as eff_ViT
from vit_pytorch import ViT as vanilla_ViT
# from vit_pytorch import SimpleViT
from vit_pytorch.distill import DistillableViT, DistillWrapper
from vit_pytorch.cvt import CvT

from linformer import Linformer
import torch.nn as nn
import timm

""" Timm model uses the same snippet of code for pre-training
    hence we define a class for pretraining models
"""

class PreTrainModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        """
        """
        self.model_name = model_name
        self.cnn = timm.create_model(self.model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.cnn.num_features

        # Used for Regression.
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.feature_dim, 1024),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        cnn_features = self.cnn(x)
        # reshape to maxpool shape req
        # print(cnn_features.size())
        cnn_features = cnn_features.view(cnn_features.size(0),1,-1)
        # print(cnn_features.size())
        predict = self.fc(cnn_features)
        predict = torch.squeeze(predict)
        return predict


""" Using Vit-Pytorch for training ViT from scratch
    Vision transformer models from 
    https://github.com/lucidrains/vit-pytorch
"""
def linformer(CONFIG, device):
    """
    CONFIG = CONFIG['Linformer_model']
    """
    
    efficient_transformer = Linformer(CONFIG['dim'],
                                      CONFIG['seq_len'],  # 7x7 patches + 1 cls-token
                                      CONFIG['depth'],
                                      CONFIG['heads'],
                                      CONFIG['k']
                                     )
    
    model = eff_ViT(dim=128,
                image_size=512,
                patch_size=CONFIG['patch_size'],#32,
                num_classes=CONFIG['num_classes'], #2,
                transformer=efficient_transformer,
                channels=3,
               ).to(device)
    
    return model


def vanilla(CONFIG, device):

    model = vanilla_ViT(
                dim=CONFIG['dim'], #Last dimension of output tensor 
                image_size=CONFIG['image_size'],
                patch_size=CONFIG['patch_size'],#32,
                num_classes=CONFIG['num_classes'], #2,
                depth=CONFIG['depth'],
                heads=CONFIG['heads'],
                channels=3,
                dropout=CONFIG['dropout'],
                pool=CONFIG['cls'],
                emb_dropout=0.1,
                mlp_dim=CONFIG['mlp_dim'],
                ).to(device)

    return model


# def simple_vit(CONFIG, device):

#     model = SimpleViT(
#         image_size=CONFIG['image_size'],
#         patch_size=CONFIG['patch_size'],
#         num_classes=CONFIG['num_classes'],
#         dim=CONFIG['dim'],
#         depth=CONFIG['depth'],
#         heads=CONFIG['heads'],
#         mlp_dim=CONFIG['mlp_dim']
#     ).to(device)

#     return model


# Models that make use of CNN
def Conv_ViT(CONFIG, device):

    model = CvT(
        num_classes = CONFIG['num_classes'],
        s1_emb_dim = 64,        # stage 1 - dimension
        s1_emb_kernel = 7,      # stage 1 - conv kernel
        s1_emb_stride = 4,      # stage 1 - conv stride
        s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
        s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
        s1_heads = 1,           # stage 1 - heads
        s1_depth = 1,           # stage 1 - depth
        s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
        s2_emb_dim = 192,       # stage 2 - (same as above)
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,       # stage 3 - (same as above)
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 4,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = CONFIG['dropout']
    ).to(device)

    return model


def distill_ViT(CONFIG):
    teacher = resnet50(pretrained = True)

    model = DistillableViT(
        image_size=CONFIG['image_size'],
        patch_size=CONFIG['patch_size'],
        num_classes=CONFIG['num_classes'],
        dim=CONFIG['dim'],
        depth=CONFIG['depth'],
        heads=CONFIG['heads'],
        mlp_dim=CONFIG['mlp_dim'],
        dropout = CONFIG['dropout'],
        emb_dropout = 0.1
    )

    distiller = DistillWrapper(
        student = model,
        teacher = teacher,
        temperature = 3,           # temperature of distillation
        alpha = 0.5,               # trade between main loss and distillation loss
        hard = False               # whether to use soft or hard distillation
    )

    return distiller, model