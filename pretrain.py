""" Pretrain.py file for traning CNN and ViT models from pretrained weights
"""

# Import statements
import torch
import pickle
import os
from data import get_data_loader
from data_utils import generate_stem_dataset
from models import PreTrainModel, ViTPreTrainModel
from train_utils import evaluate, print_msg, train
from configs.pretrain_config import *

import wandb
import argparse

parser = argparse.ArgumentParser(description='Pre-training ViTs on Fundus images')
parser.add_argument('--model', type=str, default="google/vit-base-patch16-224-in21k", help='link for the pre-trained Vit model')
parser.add_argument('--config', type=str, default="cnn", help='configuration to use')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config

# cnn_model_names = ['efficientnet_b1','densenet121', 'resnet50d']
# vit_model_names = ['vit_small_patch16_224', 'vit_tiny_patch16_224','vit_base_patch16_384']
# vit_cnn_model_names = ['vit_small_resnet26d_224', 'vit_tiny_r_s16_p8_224','vit_small_resnet50d_s16_224']

model_path = '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/'


def main(args):
    if args.config == 'cnn':
        CONFIG = CNN_CONFIG
    elif args.config == 'vit':
        CONFIG = VIT_CONFIG
    else:
        print("Configuration {} not recognized".format(args.config))
    
    # run models for a particular config
    run(args.model, CONFIG )



def run(pre_trained_model, PreTrain_CONFIG):

    batch_size = PreTrain_CONFIG['BATCH_SIZE']
    num_workers = PreTrain_CONFIG["NUM_WORKERS"]
    data_path=PreTrain_CONFIG['DATA_PATH']
    record_path = PreTrain_CONFIG['RECORD_PATH']
    input_size = PreTrain_CONFIG['INPUT_SIZE']
    data_aug=PreTrain_CONFIG['DATA_AUGMENTATION']
    # model config
    # feature_dim =PreTrain_CONFIG['FEATURE_DIM'] 
    learning_rate = PreTrain_CONFIG['LEARNING_RATE']
    epochs = PreTrain_CONFIG['EPOCHS']
    
    # save_dir = os.path.split(PreTrain_CONFIG['SAVE_PATH'])
    if not os.path.exists(PreTrain_CONFIG['SAVE_PATH']):
        os.makedirs(PreTrain_CONFIG['SAVE_PATH'])
    
    # rec_dir = os.path.split(PreTrain_CONFIG['RECORD_PATH'])
    if not os.path.exists(PreTrain_CONFIG['RECORD_PATH']):
        os.makedirs(PreTrain_CONFIG['RECORD_PATH'])



    train_dataset, test_dataset, val_dataset = generate_stem_dataset(data_path, input_size, data_aug)

    train_loader, val_loader, test_loader, weighted_sampler = get_data_loader(train_dataset, val_dataset, test_dataset, batch_size, num_workers)

    # for pre_trained_model in model_list:
    print('========================================')
    print('PreTraining For: {}'.format(pre_trained_model))
    print('========================================')
    wandb.init(project='diabetic_binary_experiment',
                job_type='train',
                name=pre_trained_model,
                dir=model_path,
                settings=wandb.Settings(start_method="fork"))

    save_path = PreTrain_CONFIG['SAVE_PATH'] + pre_trained_model +'.pt'
    record_path = PreTrain_CONFIG['RECORD_PATH'] + pre_trained_model +'.rec'
    if args.config == 'vit':
        # use vit pretrain model
        model = ViTPreTrainModel(pre_trained_model)
    else:
        model = PreTrainModel(pre_trained_model)
    model.to(device)

    # track gradients
    wandb.watch(model)

    # print model config    
    # module = model.module if isinstance(model, torch.nn.DataParallel) else model

    # print_msg('Trainable layers: ', ['{}\t{}'.format(k, v) for k, v in module.layer_configs()])

    # define loss and optimizier
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0005)
    # learning rate decay
    milestones = [5,10,15]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    record_epochs, accs, losses = train(model, train_loader, val_loader, criterion, optimizer, epochs, save_path, device,
            weighted_sampler=weighted_sampler, lr_scheduler=lr_scheduler, extra_loss=None)

    pickle.dump(
        (record_epochs, accs, losses),
        open(record_path, 'wb')
    )

    # test the network
    evaluate(save_path,device, test_loader)

    wandb.finish()


if __name__ == '__main__':
    main(args)


