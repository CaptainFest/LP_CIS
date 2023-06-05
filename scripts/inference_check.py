import os
import sys
import json
import argparse
import pandas as pd
from time import time
from tqdm import tqdm

from evaluate import load
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(1, '/nfs/home/isaitov/NL_git/NL/')
from src.dataload import IAMDataset


def get_names_and_np(folder: str):
    data = []
    for fn in os.listdir(os.path.join(folder, 'img')):
        with open(os.path.join(folder, 'ann', f"{fn.rsplit('.', 1)[0]}.json"), 'r') as f:
            js_data = json.load(f)
            data.append([fn, js_data['description']])
    return data


def get_df_from_folder(folder: str, columns=['file_name', 'text']):
    data = get_names_and_np(folder)
    df = pd.DataFrame(data, columns=columns)
    return df


def update_model_config(model):
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    return model


def model_init():
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    assert model.config.decoder.is_decoder is True
    assert model.config.decoder.add_cross_attention is True
    model = update_model_config(model)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--ocr_folder', type=str,
                        default='/nfs/home/isaitov/NL/data/autoriaNumberplateOcrRu-2021-09-01')
    args = parser.parse_args()
    return args


def save_results(results, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(f'{output_folder}/results.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':

    cer_metric = load('cer')

    args = parse_args()

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")

    ocr_folder = args.ocr_folder
    test_ocr_folder = os.path.join(ocr_folder, 'test')
    try:
        test_df = pd.read_csv(os.path.join(ocr_folder, 'test_df.csv'))
    except:
        print('Error. re-init')
        test_df = get_df_from_folder(test_ocr_folder)
        test_df.to_csv(os.path.join(ocr_folder, 'test_df.csv'), index=False)
    test_dataset = IAMDataset(root_dir=os.path.join(test_ocr_folder, 'img/'),
                              df=test_df,
                              processor=processor,
                              size=(384, 384))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = model_init()
    model.to(args.device)

    batch_timing = []
    print("Running evaluation...")
    for batch in tqdm(test_dataloader):
        start_time = time()
        # predict using generate
        pixel_values = batch["pixel_values"].to(args.device)
        outputs = model.generate(pixel_values)
        print(outputs, outputs.shape)
        # decode
        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        print(labels, pred_str, labels.shape)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        batch_timing.append(time()-start_time)
        break

    output_dir = f"../exps/bs{args.batch_size}_{args.device}_trocr_base"
    results = {'batch_size': args.batch_size, 'device': args.device,
         'instance_mean_timing': sum(batch_timing)/len(test_dataset),
         'batch_mean_timing': sum(batch_timing)/len(batch_timing)
    }
    save_results(args, output_dir)
    print(f'Results saved in: {output_dir}')
