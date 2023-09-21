import os
import sys
import json
import torch
import traceback
from evaluate import load
from typing import List, Tuple, Generator, Any, Dict

from nomeroff_net.tools.mcm import modelhub, get_device_torch
from nomeroff_net.tools.image_processing import normalize_img
from nomeroff_net.tools.ocr_tools import (StrLabelConverter,
                                          decode_prediction,
                                          decode_batch)
from nomeroff_net.nnmodels.ocr_model import NPOcrNet, weights_init
from nomeroff_net.data_modules.numberplate_ocr_data_module import OcrNetDataModule
from nomeroff_net.pipes.number_plate_text_readers.base.ocr import OCR   

sys.path.insert(1, '/nfs/home/isaitov/NL/src/')
from ocr_folders import dict_ocr_folders


class cer_OCR(OCR):
    
    @torch.no_grad()
    def cer_calc(self, dataset) -> float:
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        cer_metric = load("cer")
        for idx in range(len(dataset)):
            img, text = dataset[idx]
            img = img.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            logits = self.model(img)
            pred_text = decode_prediction(logits.cpu(), self.label_converter)
            cer_metric.add(prediction=pred_text,reference=text)
        return cer_metric.compute()
    
    def get_cer(self, data_split:str) -> float:
        generator = None
        if data_split == "train":
            generator = self.dm.train_image_generator
        elif data_split == "valid":
            generator = self.dm.val_image_generator
        elif data_split == "test":
            generator = self.dm.test_image_generator
        else:
            print(f"Wrong data split specification! {data_split}")
        cer = self.cer_calc(generator)
        print(f'{data_split} cer: {cer} in {len(self.dm.test_image_generator)}')
        return cer
    
def save_results(val_cer, test_cer, folder_path:str, split:str):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    with open(os.path.join(folder_path, f'{split}_cer.json'), 'w') as f:
        json.dump({'val_cer':val_cer, 'test_cer':test_cer}, f)
    print('Saved evaluation')
    

if __name__ == "__main__":
    for key in dict_ocr_folders:
        try:
            print(f'Evaluating {key}')
            det = cer_OCR()
            det.prepare(dict_ocr_folders['Ua'])
            det.model_name = key #.lower()
            det.load()
            val_cer = det.get_cer('valid')
            test_cer = det.get_cer('test')
            save_results(val_cer, test_cer, folder_path='exps/nomeroff_app', split=key)
            print(20*'-')
        except Exception:
            traceback.print_exc()
            print(f'Dataset {key} got error!')
