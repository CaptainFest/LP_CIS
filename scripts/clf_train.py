import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader

sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from training import fit
from siam_model import TripletNetwork, ClassificationNet
from siam_dataload import TripletDataset, SingleDataset, prepare_multilingual_OCR_dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='from_args')
    parser.add_argument('--emb_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_subsample', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--triplet_model_path', type=str, default='/nfs/home/isaitov/NL/data/siam/')
    parser.add_argument('--save_folder', type=str, default='/nfs/home/isaitov/NL/data/siam/')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    train_test_dict = {'train': '../data/train_all_OCR_df.csv', 'test': '../data/test_all_OCR_df.csv'}

    train_clf_dataset = SingleDataset(train_test_dict, train=True, train_subsample=args.train_subsample)
    test_clf_dataset = SingleDataset(train_test_dict, train=False)
    train_clf_loader = DataLoader(train_clf_dataset, batch_size=args.batch_size * 3, shuffle=True)
    test_clf_loader = DataLoader(test_clf_dataset, batch_size=args.batch_size * 3, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    triplet_model = TripletNetwork(args.emb_size)
    triplet_model.load_state_dict(torch.load(args.triplet_model_path))

    class_model = ClassificationNet(triplet_model, args.emb_size, 9)
    class_model.to(device)

    clf_loss = nn.CrossEntropyLoss()
    clf_optimizer = Adam(class_model.parameters(), lr=args.lr)
    clf_scheduler = lr_scheduler.StepLR(clf_optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 100
    fit('clf', train_clf_loader, test_clf_loader, class_model, clf_loss, clf_optimizer,
        clf_scheduler, args.epochs, device, log_interval, args.save_folder,
        args.exp_name, args.batch_size, args.emb_size)
