import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader
import lightning as l
import lightning.pytorch as pl

sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from training import fit
from losses import OnlineTripletLoss
from siam_model import TripletNetwork
from siam_dataload import SingleDataset, TripletDataset, prepare_multilingual_OCR_dataset, \
                          BalancedBatchSampler, RandomNegativeTripletSelector, HardestNegativeTripletSelector, \
                          SemihardNegativeTripletSelector
from lightning_module import LitTriplet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='from_args')
    parser.add_argument('--balanced_sampling', action=argparse.BooleanOptionalAction)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--emb_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_subsample', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--save_folder', type=str, default='/nfs/home/isaitov/NL/data/siam/')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--online', type=str, default=None)
    parser.add_argument('--margin', type=float, default=1.)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    train_test_dict = {'train': '../data/train_all_OCR_df.csv', 'test': '../data/test_all_OCR_df.csv'}

    if args.online is not None:
        train_dataset = SingleDataset(train_test_dict, train=True, train_subsample=args.train_subsample,
                                      random_state=args.seed)
        test_dataset = SingleDataset(train_test_dict, train=False)
    else:
        train_dataset = TripletDataset(train_test_dict, train=True, train_subsample=args.train_subsample,
                                       random_state=args.seed)
        test_dataset = TripletDataset(train_test_dict, train=False)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    if args.balanced_sampling:
        train_batch_sampler = BalancedBatchSampler(train_dataset.data_df, n_samples=args.n_samples)
        test_batch_sampler = BalancedBatchSampler(test_dataset.data_df, n_samples=args.n_samples)
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
        test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    if args.online is not None:
        margin = args.margin
        if args.online == 'random_negative':
            triplet_loss = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
        elif args.online == 'hardest_negative':
            triplet_loss = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
        elif args.online == 'semihard_negative':
            triplet_loss = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
        else:
            raise ValueError
    else:
        triplet_loss = nn.TripletMarginLoss()

    model = LitTriplet(last_feat_num=args.emb_size, loss_fn=triplet_loss)

    log_interval = 100

    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    #model = fit('siam', train_loader, test_loader, model, triplet_loss, optimizer,
    #            scheduler, args.epochs, device, log_interval, args.save_folder,
    #            args.exp_name, args.batch_size, args.emb_size, args.online)