import os
import gc
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from ray import tune
from evaluate import load
from torch.utils.data import DataLoader
from ray.tune.schedulers import PopulationBasedTraining
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator, \
                         Seq2SeqTrainer, Seq2SeqTrainingArguments, XLMRobertaTokenizer, \
                         ViTImageProcessor, AutoTokenizer, AutoFeatureExtractor


sys.path.insert(1, '/nfs/home/isaitov/NL/src/')

from dataload import IAMDataset
from ocr_folders import dict_ocr_folders


def get_names_and_np(folder:str):
    data = []
    for fn in os.listdir(os.path.join(folder, 'img')):
        with open(os.path.join(folder, 'ann', f"{fn.rsplit('.', 1)[0]}.json"), 'r') as f:
            js_data = json.load(f)
            data.append([fn, js_data['description']])
    return data

def get_df_from_folder(train_folder:str, val_folder:str, test_folder:str, columns=['file_name', 'text']):
    train_data, val_data, test_data = get_names_and_np(train_folder), \
                                      get_names_and_np(val_folder),   \
                                      get_names_and_np(test_folder)
    train_df, val_df, test_df = pd.DataFrame(train_data, columns=columns), \
                                pd.DataFrame(val_data, columns=columns),   \
                                pd.DataFrame(test_data, columns=columns)
    return train_df, val_df, test_df


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


def model_init(args):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder, args.decoder
    )
    assert model.config.decoder.is_decoder is True
    assert model.config.decoder.add_cross_attention is True
    model = update_model_config(model)
    return model


def compute_metrics(pred):
    cer_metric = load('cer')
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer_metric = load("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--decoder', type=str, default='bert-base-cased')
    parser.add_argument('--exp_name', type=str, default='from_args')
    parser.add_argument('--aug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--eval_steps', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--lowercase', action=argparse.BooleanOptionalAction)
    parser.add_argument('--square', action=argparse.BooleanOptionalAction)
    parser.add_argument('--ocr_folder', type=str, default='/nfs/home/isaitov/NL/data/autoriaNumberplateOcrRu-2021-09-01')
    args = parser.parse_args()
    return args

def save_args(args, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(f'{output_folder}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == '__main__':
       
    cer_metric = load('cer')
    
    args = parse_args()
    
    
    processor = TrOCRProcessor(image_processor=AutoFeatureExtractor.from_pretrained(args.encoder),
                               tokenizer=AutoTokenizer.from_pretrained(args.decoder))
    
    loop_ocr_folders = {args.ocr_folder.split('-')[0][-2:]:args.ocr_folder} if args.ocr_folder !='all' else dict_ocr_folders
    
    for key in loop_ocr_folders:
    
        ocr_folder = loop_ocr_folders[key]
        train_ocr_folder, val_ocr_folder, test_ocr_folder = os.path.join(ocr_folder, 'train'),  \
                                                            os.path.join(ocr_folder, 'val'),   \
                                                            os.path.join(ocr_folder, 'test')
        
        try:
            train_df = pd.read_csv(os.path.join(ocr_folder, 'train_df.csv'))
            val_df = pd.read_csv(os.path.join(ocr_folder, 'val_df.csv'))
            test_df = pd.read_csv(os.path.join(ocr_folder, 'test_df.csv'))
        except:
            print('Error. re-init')
            train_df, val_df, test_df = get_df_from_folder(train_ocr_folder, val_ocr_folder, test_ocr_folder)
            train_df.to_csv(os.path.join(ocr_folder, 'train_df.csv'), index=False)
            val_df.to_csv(os.path.join(ocr_folder, 'val_df.csv'), index=False)
            test_df.to_csv(os.path.join(ocr_folder, 'test_df.csv'), index=False)
            
        if args.lowercase:
            train_df['text'].apply(lambda x: x.lower())
            val_df['text'].apply(lambda x: x.lower())
            test_df['text'].apply(lambda x: x.lower())
        
        train_dataset = IAMDataset(root_dir=os.path.join(train_ocr_folder, 'img/'),
                                   df=train_df,
                                   processor=processor,
                                   train=True,
                                   aug=args.aug,
                                   square=args.square)
        val_dataset = IAMDataset(root_dir=os.path.join(val_ocr_folder, 'img/'),
                                   df=val_df,
                                   processor=processor,
                                   square=args.square)
        test_dataset = IAMDataset(root_dir=os.path.join(test_ocr_folder, 'img/'),
                                   df=test_df,
                                   processor=processor,
                                   square=args.square)
        
        output_dir = f'exps/{args.exp_name}_{key}' if args.exp_name != 'from_args' else \
                     f'exps/{args.tune}_{args.encoder}_{args.decoder}_{key}_{args.train_batch_size}_{args.max_steps}_{args.eval_steps}'
        save_args(args, output_dir)
        print(f'Args saved in: {output_dir}')
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=8,
            load_best_model_at_end=True,
            fp16=True,
            save_steps=args.eval_steps,
            eval_steps=args.eval_steps,
            max_steps=args.max_steps,
            logging_dir=f"{output_dir}/logs",
            logging_steps=args.eval_steps
        )
        
        model = model_init(args)

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=processor.feature_extractor,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
        )
        
        print("Running evaluation...")
        
        trainer.train()
        
        predict_metrics = trainer.predict(test_dataset, metric_key_prefix='test')
        print(predict_metrics.metrics)
        with open(f'{output_dir}/logs/test_results.json', 'w') as f:
            json.dump(predict_metrics.metrics, f, indent=2)
        
        trainer.save_model(output_dir)
        # free memory
        try: 
            trainer.model.to('cpu')
        except:
            pass
        del trainer
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        