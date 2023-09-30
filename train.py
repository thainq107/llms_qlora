import os
import json
import argparse
import logging
import sys
from functools import partial
import numpy as np

import datasets
from datasets import DatasetDict

import evaluate
import bitsandbytes as bnb

import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    GenerationConfig,
    AutoTokenizer,
    LlamaTokenizer,
    Seq2SeqTrainer,
    set_seed,
)

from utils import (
    get_last_checkpoint, 
    smart_tokenizer_and_embedding_resize, 
    print_trainable_parameters, 
    get_gpu_utilization,
)
from model.dataloader import load_dataset_from_path, preprocess_function
from model.model import get_accelerate_model
from model.metric import load_metric, seq2seq_compute_metrics
from model.callback import SavePeftModelCallback

from huggingface_hub import login, create_repo, delete_repo

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def train(model_args, data_args, training_args, generation_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)
    set_seed(args.seed)
    # login hub
    if training_args.push_to_hub:
        login(
            token=training_args.hub_token
        )
        try:
            create_repo(training_args.hub_model_id, private=False)
        except:
            delete_repo(training_args.hub_model_id)
            create_repo(training_args.hub_model_id, private=False)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    # load dataset
    raw_dataset = load_dataset_from_path(
        data_args.save_data_dir,
        data_args.dataset_name,
        data_args.train_file, 
        data_args.validation_file,
        data_args.test_file
    )
    raw_dataset = DatasetDict(raw_dataset)
    logger.info(f"Dataset loaded: {raw_dataset}")

     # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
    )
    model = get_accelerate_model(args, checkpoint_dir)
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })

    model.config.use_cache = False
    print('loaded model')

    # prepare metric
    metric = load_metric(data_args.metric_name)
    compute_metrics = seq2seq_compute_metrics(tokenizer, metric)

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="Dataset map pre-processing"):
        processed_dataset = raw_dataset.map(
            partial(
                preprocess_function,
                data_args=data_args,
                tokenizer=tokenizer
            ),
            batched=True,
            load_from_cache_file=False,
            remove_columns=['sentence', 'label'],
            desc="Running tokenizer on dataset",
        )
    
    # ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=processed_dataset["test"],
        eval_dataset=processed_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(processed_dataset["train"])
        metrics["gpu_memory"] = get_gpu_utilization()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            eval_dataset=processed_dataset["validation"],
            metric_key_prefix="eval"
        )
        metrics["eval_samples"] = len(processed_dataset["validation"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        if "labels" in processed_dataset["test"].features:
            metrics = trainer.evaluate(eval_dataset=processed_dataset["test"])
            metrics["test_samples"] = len(processed_dataset["test"])
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
            all_metrics.update(metrics)
        
        predictions = trainer.predict(processed_dataset["test"], metric_key_prefix="predict").predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predictions = [str(pred).strip() for pred in predictions]
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        with open(output_predict_file, "w") as writer:
            writer.write("\n".join(predictions))
        logger.info("Predict results saved at {}".format(output_predict_file))


    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
    
    # Save processor and create model card
    tokenizer.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.create_model_card()
        trainer.push_to_hub()