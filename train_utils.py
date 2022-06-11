""" Train and Evaluate scripts 
"""
import numpy as np
import torch
import wandb
from tqdm import tqdm
from metrics import classify, accuracy, quadratic_weighted_kappa



def train(model, train_loader, val_loader, loss_function, optimizer, epochs, save_path, device,
          weighted_sampler=None, lr_scheduler=None, extra_loss=None):
    max_kappa = 0
    record_epochs, accs, losses = [], [], []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        correct = 0
        total = 0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X = train_data['pixel_values']
            y = train_data['labels']

            X, y = X.to(device), y.float().to(device)
            

            # forward
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            if extra_loss:
                loss += extra_loss(model, X, y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            total += y.size(0)
            correct += accuracy(y_pred, y) * y.size(0)
            avg_loss = epoch_loss / (step + 1)
            avg_acc = correct / total
            # if step % 200 == 0: # print every 200 steps
            #     progress.set_description(
            #         'epoch: {}, loss: {:.6f}, acc: {:.4f}'
            #         .format(epoch, avg_loss, avg_acc)
            #     )


        # save model
        c_matrix = np.zeros((5, 5), dtype=int)
        acc = _eval(model, device, val_loader, c_matrix)
        kappa = quadratic_weighted_kappa(c_matrix)
        print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
        if kappa > max_kappa:
            torch.save(model, save_path)
            max_kappa = kappa
            print_msg('Model save at {}'.format(save_path))

        # wandblogging
        wandb.log({"epoch": epoch, "Loss": avg_loss})
        wandb.log({"epoch": epoch, "Accuracy": acc }) #reporting validation accuracy
        # record
        record_epochs.append(epoch)
        accs.append(acc)
        losses.append(avg_loss)

        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()

        # learning rate update
        if lr_scheduler:
            lr_scheduler.step()
            if epoch in lr_scheduler.milestones:
                print_msg('Learning rate decayed to {}'.format(lr_scheduler.get_lr()[0]))

    print('Best validation accuracy: {}'.format(max_kappa))
    return record_epochs, accs, losses


def _eval(model, device, dataloader, c_matrix=None):
    # 
    model.eval()
    torch.set_grad_enabled(False)

    correct = 0
    total = 0
    progress = tqdm(enumerate(dataloader))
    for step, test_data in progress:
        X = test_data['pixel_values']
        y = test_data['labels']

        X, y = X.to(device), y.float().to(device)

        y_pred = model(X)
        total += y.size(0)
        correct += accuracy(y_pred, y, c_matrix) * y.size(0)
    acc = round(correct / total, 4)

    model.train()
    torch.set_grad_enabled(True)
    return acc


def evaluate(model_path, device, test_loader):
    c_matrix = np.zeros((5, 5), dtype=int)

    trained_model = torch.load(model_path, map_location=torch.device(device))
    test_acc = _eval(trained_model, device, test_loader, c_matrix)
    q_kappa = quadratic_weighted_kappa(c_matrix)
    #Log
    wandb.log({"Test Accuracy": test_acc})
    wandb.log({"Kappaq":q_kappa})
    wandb.log({"Conf_Matrix": c_matrix})

    print('========================================')
    print('Finished! test acc: {}'.format(test_acc))
    print('Confusion Matrix:')
    print(c_matrix)
    print('quadratic kappa: {}'.format(q_kappa))
    print('========================================')


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)