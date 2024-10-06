

from libs.classifiers.base import BaseClassifier
from libs.data_processor import DATA_PROCESSOR
from libs.utils import evaluate
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoModelForSeq2SeqLM,
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          Trainer
                          )
from datasets import Dataset
import torch
from tqdm import tqdm
import pandas as pd
from peft import get_peft_model, prepare_model_for_kbit_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 1, 
                        temperature = 0.001,
                       )
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("=")[-1]
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")
    return y_pred

def preprocess_function(row, tokenizer):
    model_inputs = tokenizer(row["text"], padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(row["sentiment"], padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs



def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

class T5Classifier(BaseClassifier):
    """
    T5Classifier is a specialized classifier that leverages the T5 model for sequence-to-sequence learning tasks. 
    This class provides methods to build, train, and evaluate the T5 model.
    Attributes:
        model (transformers.PreTrainedModel): The T5 model used for classification.
        method_name (str): The name of the method, default is "flan-t5".
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the T5 model.
        model_name (str): The name of the pre-trained T5 model, default is "google/flan-t5-large".
        maxlen (int): The maximum length of the input sequences, default is 128.
    Methods:
        __init__(maxlen=128):
            Initializes the T5Classifier with the specified maximum sequence length.
        build_model(**kwargs):
            Builds the T5 model and tokenizer with specified configurations.
        print():
            Prints a summary of the T5 model.
        train(data_processor: DATA_PROCESSOR, peft_config, **kwargs):
            Fine-tunes the T5 model using the provided training data and configurations.
        evaluate(data_processor: DATA_PROCESSOR, name: str):
            Evaluates the T5 model using the provided test data and evaluation method.
    """
    def __init__(self, maxlen = 128):
        self.model = None
        self.method_name = "flan-t5"
        self.tokenizer = None
        self.model_name = "google/flan-t5-large"
        self.maxlen = maxlen
        
    def build_model(self, **kwargs):
        
        compute_dtype = getattr(torch, "float16")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path = self.model_name,
            device_map=device,
            torch_dtype=compute_dtype,
            quantization_config=bnb_config, 
            low_cpu_mem_usage=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.model, self.tokenizer = setup_chat_format(model, tokenizer)
        
    def print(self):
        print(self.model.summary())
    

    def train(self, data_processor: DATA_PROCESSOR, peft_config, **kwargs):
        
        #####  FINETUNING MODEL  #########
        train_data = Dataset.from_pandas(data_processor.X_train)
        eval_data = Dataset.from_pandas(data_processor.X_eval)

        tokenized_train_dataset = train_data.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
        tokenized_eval_dataset = eval_data.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text', 'sentiment'])
        tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(['text', 'sentiment'])
        
        output_dir="t5_trained_weigths"

        training_arguments = TrainingArguments(
            output_dir=output_dir,                    # directory to save and repository id
            num_train_epochs=3,                       # number of training epochs
            per_device_train_batch_size=1,            # batch size per device during training
            gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
            gradient_checkpointing=True,              # use gradient checkpointing to save memory
            optim="paged_adamw_32bit",
            save_steps=0,
            logging_steps=25,                         # log every 10 steps
            learning_rate=2e-4,                       # learning rate, based on QLoRA paper
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
            max_steps=-1,
            warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
            group_by_length=True,
            lr_scheduler_type="cosine",               # use cosine learning rate scheduler
            report_to="tensorboard",                  # report metrics to tensorboard
            evaluation_strategy="epoch"               # save checkpoint every epoch
        )
        self.model = prepare_model_for_kbit_training(self.model)
        peft_model = get_peft_model(self.model, 
                            peft_config)
        
        print(print_number_of_trainable_model_parameters(peft_model))

        trainer = Trainer(
            model=peft_model,
            args=training_arguments,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            #peft_config=peft_config,
            #dataset_text_field="text",
            # tokenizer=self.tokenizer,
            # max_seq_length=1024,
            # packing=False,
            # dataset_kwargs={
            #     "add_special_tokens": False,
            #     "append_concat_token": False,
            # }
        )
        # trainer = Trainer(
        #     model=self.model,
        #     args=training_arguments,
        #     train_dataset=train_data,
        #     eval_dataset=eval_data,
        #     peft_config=peft_config,
        #     dataset_text_field="text",
        #     tokenizer=self.tokenizer,
        #     max_seq_length=1024,
        #     packing=False,
        #     dataset_kwargs={
        #         "add_special_tokens": False,
        #         "append_concat_token": False,
        #     }
        # )
        print("Start training .....")
        trainer.train()
        print("Saving models ....")
        # Save trained model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
    def evaluate(self, data_processor: DATA_PROCESSOR, name : str):
        y_pred = predict(data_processor.X_test, self.model, self.tokenizer)
        y_true = data_processor.y_test_true
        evaluate(y_true, y_pred, method=name)