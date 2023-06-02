import os
import json
import argparse
from glob import glob
import matplotlib.pyplot as plt

from evaluate import load
from mmocr.apis import MMOCRInferencer
from torch.utils.data import Dataset, DataLoader

def get_json(json_path:str) -> str:
    with open(json_path, 'r') as f:
        json_data = json.load(f)['description']
        return json_data


def get_preds(model_preds:dict) -> list:
    res_list = list(map(lambda x: x['rec_texts'][0], res['predictions']))
    return res_list


def lower_list(data:list):
    return list(map(lambda x: x.lower(), data))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/autoriaNumberplateOcrRu-Military-2022-03-25/train')
    args = parser.parse_args()
    return args

class ImgNamesDataset(Dataset):
    def __init__(self, data_path:str):
        self.data_path = data_path
        self.img_names = os.listdir(os.path.join(self.data_path, 'img'))

    def __getitem__(self, item):
        img_path = os.path.join(self.data_path, 'img', self.img_names[item])
        json_data = get_json(os.path.join(self.data_path, 'ann', self.img_names[item].split('.', -1)[0]+'.json'))
        return img_path, json_data

    def __len__(self):
        return len(self.img_names)


RETURN_VIS = False
VERBOSE = False
BATCH_SIZE = 100
LOWER = True


if __name__ == '__main__':
    args = get_args()
    cer = load('cer')
    ocr = MMOCRInferencer(det=None, rec='SAR')
    dataset_folder = args.data_path

    training_data = ImgNamesDataset(dataset_folder)
    dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)

    for img_names_batch, json_data_batch in dataloader:
        res = ocr(img_names_batch, return_vis=RETURN_VIS)
        preds = get_preds(res)
        if LOWER:
            preds = lower_list(preds)
            json_data_batch = lower_list(json_data_batch)

        cer.add_batch(predictions=preds, references=json_data_batch)
        if VERBOSE:
            print(preds, json_data_batch)

    if RETURN_VIS:
        plt.imshow(res['visualization'][0])
    print(cer.compute())
