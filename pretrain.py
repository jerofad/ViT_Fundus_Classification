""" Pretrain.py file for traning CNN and ViT models from pretrained weights
"""

# Import statements
import torch
import pickle
import os
from data import get_data_loader
from data_utils import generate_stem_dataset
from models import PreTrainModel
from train_utils import evaluate, print_msg, train
from configs.pretrain_config import *

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config

cnn_model_names = ['efficientnet_b1','densenet121', 'resnet50d']
vit_model_names = ['vit_small_patch16_224', 'vit_tiny_patch16_224','vit_base_patch16_384']
vit_cnn_model_names = ['vit_small_resnet26d_224', 'vit_tiny_r_s16_p8_224','vit_small_resnet50d_s16_224']
model_path = '/mnt/qb/berens/users/jfadugba97/RetinaClassification/result/ViTs/'

# Dataloader

def main():
    # run models for CNN pre_train
    run(cnn_model_names, CNN_CONFIG)
    # run models for ViT pre_train
    run(vit_model_names, VIT_CONFIG)
    # run models for ViT_+CNN pre_train
    run(vit_cnn_model_names, VIT_CONFIG)



def run(model_list, PreTrain_CONFIG):

    batch_size = PreTrain_CONFIG['BATCH_SIZE']
    num_workers = PreTrain_CONFIG["NUM_WORKERS"]
    data_path=PreTrain_CONFIG['DATA_PATH']
    record_path = PreTrain_CONFIG['RECORD_PATH']
    input_size = PreTrain_CONFIG['INPUT_SIZE']
    data_aug=PreTrain_CONFIG['DATA_AUGMENTATION']
    # model config
    feature_dim =PreTrain_CONFIG['FEATURE_DIM'] 
    learning_rate = PreTrain_CONFIG['LEARNING_RATE']
    epochs = PreTrain_CONFIG['EPOCHS']
    
    save_dir = os.path.split(PreTrain_CONFIG['SAVE_PATH'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    rec_dir = os.path.split(PreTrain_CONFIG['RECORD_PATH'])
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)



    train_dataset, test_dataset, val_dataset = generate_stem_dataset(data_path, input_size, data_aug)

    train_loader, val_loader, weighted_sampler = get_data_loader(train_dataset, val_dataset, batch_size, num_workers)


    for pre_trained_model in model_list:

        wandb.init(project='diabetic_binary_experiment',
                    job_type='train',
                    name=pre_trained_model,
                    dir=model_path)

        save_path = PreTrain_CONFIG['SAVE_PATH'] + pre_trained_model +'.pt'
        record_path = PreTrain_CONFIG['RECORD_PATH'] + pre_trained_model +'.rec'
        model = PreTrainModel(pre_trained_model, feature_dim)
        model.to(device)

        # track gradients
        wandb.watch(model)

        # print model config    
        module = model.module if isinstance(model, torch.nn.DataParallel) else model

        print_msg('Trainable layers: ', ['{}\t{}'.format(k, v) for k, v in module.layer_configs()])

        # define loss and optimizier
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0005)
        # learning rate decay
        milestones = [150, 220]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        record_epochs, accs, losses = train(model, train_loader, val_loader, criterion, optimizer, epochs, save_path, device,
                weighted_sampler=weighted_sampler, lr_scheduler=lr_scheduler, extra_loss=None)

        pickle.dump(
            (record_epochs, accs, losses),
            open(record_path, 'wb')
        )

        # test the stem network
        evaluate(save_path, test_dataset, num_workers)

        wandb.finish()






if __name__ == '__main__':
    main()


