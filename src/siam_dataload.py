import os
import numpy as np
import pandas as pd
from PIL import Image
from itertools import combinations

import torch
from torch.utils.data import Dataset, BatchSampler
from torchvision.transforms import Resize, Compose, ToTensor, ToPILImage

from ocr_folders import dict_ocr_folders, dict_win_ocr_folders


from utils import pdist


def prepare_multilingual_OCR_dataset(data_folders: dict, train: str) -> pd.DataFrame:
    data_arr = np.array([[], [], [], []])
    for i, key_country in enumerate(data_folders):
        data_folder_path = os.path.join(data_folders[key_country], train, 'img')
        filenames = os.listdir(data_folder_path)
        data_arr = np.hstack((data_arr, [[data_folder_path]*len(filenames),
                                         filenames,
                                         [key_country]*len(filenames),
                                         [i]*len(filenames)]))
    dataset_df = pd.DataFrame(np.swapaxes(data_arr, 0, 1), columns=['data_folder_path', 'filename',
                                                                    'reg_name', 'reg_label'])
    return dataset_df


def get_multilingual_OCR_dataset(train_test_dict, train: str = 'train', system: str = 'linux'):
    try:
        data_all_OCR_df = pd.read_csv(train_test_dict[train])
    except:
        print(f'File {train_test_dict[train][:-4]} not exists. Preparing data ...')
        dict_ocr = dict_ocr_folders if system == 'linux' else dict_win_ocr_folders
        data_all_OCR_df = prepare_multilingual_OCR_dataset(dict_ocr, train=train)
        data_all_OCR_df.to_csv(train_test_dict[train], index=False)
    return data_all_OCR_df


def get_positive_data(data, i):
    ex = data[data['reg_label'] == data.iloc[i]['reg_label']].sample()
    return os.path.join(ex['data_folder_path'].item(), ex['filename'].item())


def get_negative_data(data, i):
    ex = data[data['reg_label'] != data.iloc[i]['reg_label']].sample()
    return os.path.join(ex['data_folder_path'].item(), ex['filename'].item())


def class_2_indexes(data_df, classes) -> dict:
    return {class_: data_df.loc[data_df['reg_label'] == class_].index.to_numpy() for class_ in classes}


class TripletDataset(Dataset):
    def __init__(self, train_test_dict, train=True, aug=False, size=(112, 224),
                 train_subsample=None, random_state=None, system: str = 'linux'):
        self.transform = Compose([Resize(size), ToTensor()])
        self.aug = aug
        self.train = train
        if self.train:
            self.data_df = get_multilingual_OCR_dataset(train_test_dict, train='train', system=system)
            if train_subsample is not None:
                self.data_df = self.data_df.sample(frac=train_subsample)
                per_class_num = int(len(self.data_df) * train_subsample)
                self.data_df = pd.concat([self.data_df[self.data_df['reg_label'] == cl].sample(n=per_class_num,
                                                                                               random_state=random_state)
                                          if (self.data_df[self.data_df['reg_label'] == cl].shape[0] > per_class_num)
                                          else (self.data_df[self.data_df['reg_label'] == cl])
                                          for cl in set(self.data_df['reg_label'])])
        else:
            self.data_df = get_multilingual_OCR_dataset(train_test_dict, train='test', system=system)
            self.test_triplets = self.prepare_test_triplets()
        self.data_df.reset_index(drop=True, inplace=True)
        self.labels = np.unique(self.data_df['reg_label'])

    def __len__(self):
        return len(self.data_df)

    def prepare_test_triplets(self):
        return [[os.path.join(self.data_df.iloc[i]['data_folder_path'], self.data_df.iloc[i]['filename']),
                 get_positive_data(self.data_df, i),
                 get_negative_data(self.data_df, i)]
                for i in range(len(self.data_df))]

    def __getitem__(self, idx):

        if self.train:
            ex_1 = self.data_df.iloc[idx]
            img_1_path = os.path.join(ex_1['data_folder_path'], ex_1['filename'])
            img_2_path = get_positive_data(self.data_df, idx)
            img_3_path = get_negative_data(self.data_df, idx)
        else:
            img_1_path = self.test_triplets[idx][0]
            img_2_path = self.test_triplets[idx][1]
            img_3_path = self.test_triplets[idx][2]

        # if self.train and self.aug:
        #    image = self.transform(image)

        img1 = Image.open(img_1_path).convert('RGB')
        img2 = Image.open(img_2_path).convert('RGB')
        img3 = Image.open(img_3_path).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3


class SingleDataset(Dataset):
    def __init__(self, train_test_dict, train, train_subsample=None,  size=(112, 224), random_state=None,
                 system: str = 'linux'):
        self.train = train
        self.transform = Compose([Resize(size), ToTensor()])
        if self.train:
            self.data_df = get_multilingual_OCR_dataset(train_test_dict, train='train', system=system)
            if train_subsample is not None:
                self.data_df = self.data_df.sample(frac=train_subsample)
                per_class_num = int(len(self.data_df) * train_subsample)
                self.data_df = pd.concat([self.data_df[self.data_df['reg_label'] == cl].sample(n=per_class_num,
                                                                                               random_state=random_state)
                                          if (self.data_df[self.data_df['reg_label'] == cl].shape[0] > per_class_num)
                                          else (self.data_df[self.data_df['reg_label'] == cl])
                                          for cl in set(self.data_df['reg_label'])])
        else:
            self.data_df = get_multilingual_OCR_dataset(train_test_dict, train='test', system=system)
        self.data_df.reset_index(drop=True, inplace=True)
        self.labels = np.unique(self.data_df['reg_label'])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # get file name + text
        img_path = os.path.join(*self.data_df.loc[idx, ['data_folder_path', 'filename']].values)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.data_df.loc[idx, 'reg_label']
        return img, label


class BalancedBatchSampler(BatchSampler):
    def __init__(self, data_df, n_samples):
        self.data_df = data_df
        self.labels_set = list(set(self.data_df['reg_label']))
        self.indexes_per_class = class_2_indexes(data_df, self.labels_set)
        for l in self.labels_set:
            np.random.shuffle(self.indexes_per_class[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_samples = n_samples
        self.n_classes = len(self.labels_set)
        self.n_dataset = len(self.data_df)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.indexes_per_class[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.indexes_per_class[class_]):
                    np.random.shuffle(self.indexes_per_class[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector:
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])),
                                                            torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=hardest_negative,
                                           cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=random_hard_negative,
                                           cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=lambda x: semihard_negative(x, margin),
                                           cpu=cpu)
