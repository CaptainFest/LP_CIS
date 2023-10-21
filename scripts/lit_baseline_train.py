import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader

sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from training import fit
from siam_model import EmbedNetwork
from siam_dataload import SingleDataset, prepare_multilingual_OCR_dataset
from lightning_module import BaseClf


def save_hps2logger(logger, args):
    logger.log_hyperparams(vars(args))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='from_args')
    parser.add_argument('--classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--train_subsample', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--save_folder', type=str, default='/nfs/home/isaitov/NL/data/siam/')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logs_path', type=str, default="../exps/light_logs/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    train_test_dict = {'train': '../data/train_all_OCR_df.csv', 'test': '../data/test_all_OCR_df.csv'}

    train_dataset = SingleDataset(train_test_dict, train=True, train_subsample=args.train_subsample,
                                      random_state=args.seed)
    test_dataset = SingleDataset(train_test_dict, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    emb_net = EmbedNetwork(args.classes)
    model = BaseClf(emb_net, args.classes, args.save_folder, args.exp_name)

    comet_logger = pl_loggers.CometLogger(
        api_key='fdnVusaeA1HEamDdLRAIQH6xW',
        project_name="region-recognition",
        workspace="captainfest",
        experiment_name=args.exp_name,
        save_dir=args.logs_path
    )

    save_hps2logger(comet_logger, args)
    trainer = pl.Trainer(max_epochs=args.epochs, logger=comet_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    #log_interval = 100
    #fit('clf', train_clf_loader, test_clf_loader, class_model, clf_loss, clf_optimizer,
    #    clf_scheduler, args.epochs, device, log_interval, args.save_folder,
    #    args.exp_name, args.batch_size, args.emb_size, None)
