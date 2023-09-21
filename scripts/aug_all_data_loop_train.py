import os
import gc
import json
import argparse
import numpy as np
import pandas as pd

from ray import tune
from evaluate import load
from ray.tune.schedulers import PopulationBasedTraining
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator, \
                         Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.dataload import IAMDataset


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


def model_init(trial):
    model = VisionEncoderDecoderModel.from_pretrained(
        'microsoft/trocr-base-stage1')
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


if __name__ == '__main__':
    
    dict_ocr_folders = {'Am': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrAm-2022-11-21-all/',
                        'Eu': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrEu-2023-04-25/',
                        'Kg': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrKg-2022-11-30/',
                        'Kz': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrKz-2022-11-29/',
                        'Su': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrSu-2023-03-10/',
                        'Ge': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrGe-2022-11-29/',
                        'Ua': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrUa-2023-04-18/',
                        'Md': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrMd-2023-01-27/',
                        'By': '/nfs/home/isaitov/NL/data/autoriaNumberplateOcrBy-2021-08-27/'}
    
    cer_metric = load('cer')
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    
    for key in dict_ocr_folders:
    
        ocr_folder = dict_ocr_folders[key]
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
        
        train_dataset = IAMDataset(root_dir=os.path.join(train_ocr_folder, 'img/'),
                                   df=train_df,
                                   processor=processor)
        val_dataset = IAMDataset(root_dir=os.path.join(val_ocr_folder, 'img/'),
                                   df=val_df,
                                   processor=processor)
        test_dataset = IAMDataset(root_dir=os.path.join(test_ocr_folder, 'img/'),
                                   df=test_df,
                                   processor=processor)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=f'exps/aug_{key}',
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=8,
            load_best_model_at_end=True,
            fp16=True,
            logging_steps=2,
            save_steps=1000,
            eval_steps=200,
            max_steps=1500
        )

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=None,
            tokenizer=processor.feature_extractor,
            model_init=model_init,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
        )
        
        print("Running evaluation...")
        
        trainer.train()
        
        # test eval
        for batch in tqdm(test_dataloader):
            # predict using generate
            pixel_values = batch["pixel_values"].to(device)
            outputs = model.generate(pixel_values)

            # decode
            pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(labels, skip_special_tokens=True)

            # add batch to metric
            cer_metric.add_batch(predictions=pred_str, references=label_str)

        final_score = cer_metric.compute()
        print(f"Character error rate on test {key} set:", final_score)
        
        # free memory
        try: 
            trainer.model.to('cpu')
        except:
            pass
        del trainer
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        