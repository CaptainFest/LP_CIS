import os
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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ocr_folder', type=str, default='/nfs/home/isaitov/NL/data/autoriaNumberplateOcrRu-2021-09-01')
    args = parser.parse_args()
    return args

def get_names_and_np(folder:str):
    data = []
    for fn in os.listdir(os.path.join(folder, 'img')):
        with open(os.path.join(folder, 'ann', f"{fn.rsplit('.', 1)[0]}.json"), 'r') as f:
            js_data = json.load(f)
            data.append([fn, js_data['description']])
    return data

def get_df_from_older(train_folder:str, val_folder:str, test_folder:str, columns=['file_name', 'text']):
    train_data, val_data, test_data = get_names_and_np(train_folder), \
                                      get_names_and_np(val_folder),   \
                                      get_names_and_np(test_folder)
    train_df, val_df, test_df = pd.DataFrame(train_data, columns=columns), \
                                pd.DataFrame(val_data, columns=columns),   \
                                pd.DataFrame(test_data, columns=columns)
    return train_df, val_df, test_df

def ray_hp_space(trial):
    return {
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 1,
    }

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
    
    args = parse_args()
    
    train_ocr_folder, val_ocr_folder, test_ocr_folder = os.path.join(args.ocr_folder, 'train'),  \
                                                                 os.path.join(args.ocr_folder, 'val'),   \
                                                                 os.path.join(args.ocr_folder, 'test')
    
    try:
        train_df = pd.read_csv('train_df.csv')
        val_df = pd.read_csv('val_df.csv')
        test_df = pd.read_csv('test_df.csv')
    except:
        print('Error. re-init')
        train_df, val_df, test_df = get_df_from_folder(train_ocr_folder, val_ocr_folder, test_ocr_folder)
        
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    train_dataset = IAMDataset(root_dir=os.path.join(train_ocr_folder, 'img/'),
                               df=train_df,
                               processor=processor)
    val_dataset = IAMDataset(root_dir=os.path.join(val_ocr_folder, 'img/'),
                               df=val_df,
                               processor=processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='tune_aug_exp_1',
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
    
    trainer = Seq2SeqTrainer(
        model=None,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,   # mb add small
        eval_dataset=val_dataset,      # mb add small
        model_init=model_init,
        data_collator=default_data_collator,
    )
    
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_cer",
        mode="min",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.choice([0., 0.05, 0.1]),
            # "num_beams": tune.choice([2,4,6]),
            "warmup_ratio": tune.uniform(0, 0.5),
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            # "layernorm_embedding": tune.choice([True, False]),
            "max_steps": tune.randint(500, 1500)
        },
    )

    best_trial = trainer.hyperparameter_search(
        # direction="maximize",
        backend="ray",
        hp_space=ray_hp_space,
        n_trials=10,
        keep_checkpoints_num=1,
        scheduler=scheduler,
        local_dir="~/ray_results/",
        name="tune_aug_transformer_pbt",
        log_to_file=True,
        #resume=True,
        verbose=1
        # compute_objective=compute_objective,
    )
    
    #trainer.train()