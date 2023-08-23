import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, ToPILImage

from src.ocr_folders import dict_ocr_folders


train_test_dict = {'train': '../data/train_all_OCR_df.csv', 'test': '../data/test_all_OCR_df.csv'}


def prepare_multilingual_OCR_dataset(data_folders:dict, train:str) -> pd.DataFrame:
    data_arr = np.array([[],[],[],[]])
    for i, key_country in enumerate(data_folders):
        data_folder_path = os.path.join(data_folders[key_country], train, 'img')
        filenames = os.listdir(data_folder_path)
        data_arr = np.hstack((data_arr, [[data_folder_path]*len(filenames),
                                       filenames,
                                       [key_country]*len(filenames),
                                       [i]*len(filenames)]))
    dataset_df = pd.DataFrame(np.swapaxes(data_arr,0,1), columns=['data_folder_path', 'filename', 'reg_name', 'reg_label'])
    return dataset_df


def get_multilingual_OCR_dataset(train_test_dict, train:str='train'):
    try:
        data_all_OCR_df = pd.read_csv(train_test_dict[train])
    except:
        print(f'File {train_test_dict[train][:-4]} not exists. Preparing data ...')
        data_all_OCR_df = prepare_multilingual_OCR_dataset(dict_ocr_folders, train=train)
    return data_all_OCR_df


def get_positive_data(data, i):
    ex = data[data['reg_label'] == data.iloc[i]['reg_label']].sample()
    return os.path.join(ex['data_folder_path'].item(), ex['filename'].item())


def get_negative_data(data, i):
    ex = data[data['reg_label'] != data.iloc[i]['reg_label']].sample()
    return os.path.join(ex['data_folder_path'].item(), ex['filename'].item())


class TripletDataset(Dataset):
    def __init__(self, train_test_dict, train=True, aug=False, size=(112, 224), train_subsample=None):
        self.transform = Compose([Resize(size), ToTensor()])
        self.aug = aug
        self.train = train
        if self.train:
            self.data_df = get_multilingual_OCR_dataset(train_test_dict, train='train')
            if train_subsample is not None:
                self.data_df = self.data_df.sample(frac=train_subsample)
        else:
            self.data_df = get_multilingual_OCR_dataset(train_test_dict, train='test')
            self.test_triplets = self.prepare_test_triplets()
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
        return (img1, img2, img3)
