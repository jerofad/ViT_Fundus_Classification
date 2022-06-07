import torch
from torch.utils.data import DataLoader
from data_utils import ScheduledWeightedSampler, my_collate
# create dataloader


def get_data_loader(train_dataset, val_dataset, batch_size, num_workers):

    train_targets = [sampler[1] for sampler in train_dataset.samples]
    weighted_sampler = ScheduledWeightedSampler(len(train_dataset), train_targets, replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=weighted_sampler,
        num_workers=num_workers,
        collate_fn=my_collate,
        drop_last=True
    )

    # avoid IndexError when use multiple gpus
    val_batch_size = batch_size if len(val_dataset) % batch_size >= 2 * torch.cuda.device_count() else batch_size - 2 
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_loader, val_loader, weighted_sampler


def get_test_loader(test_dataset, batch_size, num_workers):

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return test_loader

