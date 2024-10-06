

from libs.classifiers.base import BaseClassifier
from libs.data_processor import DATA_PROCESSOR
from libs.utils import evaluate
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          )
from peft import AutoPeftModelForCausalLM
from datasets import Dataset
import torch
from tqdm import tqdm
import numpy as np
from libs.data_processor import generate_text

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

class LLAMA2Classifier(BaseClassifier):
    """
    A classifier utilizing the LLAMA2 model for text classification tasks.
    Attributes:
        model (AutoModelForCausalLM): The LLAMA2 model used for classification.
        method_name (str): The name of the method, set to "llama2".
        tokenizer (AutoTokenizer): The tokenizer for the LLAMA2 model.
        model_name (str): The name of the pretrained model.
        pipe (pipeline): The pipeline for text generation.
    Methods:
        __init__():
            Initializes the LLAMA2Classifier with default attributes.
        build_model(**kwargs):
            Builds the LLAMA2 model with specified configurations.
        print():
            Prints the model architecture.
        train(data_processor: DATA_PROCESSOR, peft_config, **kwargs):
            Fine-tunes the LLAMA2 model using the provided training data and configurations.
        evaluate(data_processor: DATA_PROCESSOR, name: str):
            Evaluates the model on the test data and prints the evaluation metrics.
        merge_finetuned_model(finetuned_model_path):
            Merges the fine-tuned model with the base model and sets up the text generation pipeline.
        predict(text: str):
            Generates predictions for the given text using the text generation pipeline.
    """
    def __init__(self):
        self.model = None
        self.method_name = "llama2"
        self.tokenizer = None
        self.model_name = "NousResearch/Llama-2-7b-chat-hf"
        self.pipe = None
        
    def build_model(self, **kwargs):
        
        compute_dtype = getattr(torch, "float16")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = self.model_name,
            device_map=device,
            torch_dtype=compute_dtype,
            quantization_config=bnb_config, 
            low_cpu_mem_usage=True,
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = self.model_name, 
                                                trust_remote_code=True,
                                                )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.model, self.tokenizer = setup_chat_format(model, tokenizer)
        
    def print(self):
        print(self.model)
        
    def train(self, data_processor: DATA_PROCESSOR, peft_config, **kwargs):
        
        #####  FINETUNING MODEL  #########
        
        train_data = Dataset.from_pandas(data_processor.X_train)
        eval_data = Dataset.from_pandas(data_processor.X_eval)

        output_dir="trained_weigths"

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

        trainer = SFTTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            max_seq_length=1024,
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            }
        )
        print("Start training .....")
        trainer.train()
        print("Saving models ....")
        # Save trained model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
    def evaluate(self, data_processor: DATA_PROCESSOR, name : str, show = False):
        y_pred = predict(data_processor.X_test, self.model, self.tokenizer)
        y_true = data_processor.y_test_true
        mapping = {'positive': 2, 'neutral': 1, 'none':1, 'negative': 0}
        def map_func(x):
            return mapping.get(x, 1)
        
        y_true = np.vectorize(map_func)(y_true)
        y_pred = np.vectorize(map_func)(y_pred)
        evaluate(y_true, y_pred, method=name, show = show)
        
    def merge_finetuned_model(self, finetuned_model_path):
        compute_dtype = getattr(torch, "float16")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoPeftModelForCausalLM.from_pretrained(
            finetuned_model_path,
            torch_dtype=compute_dtype,
            return_dict=False,
            low_cpu_mem_usage=True,
            device_map=device,
        )

        merged_model = model.merge_and_unload()
        pipe = pipeline(task="text-generation", 
                            model=merged_model, 
                            tokenizer=tokenizer, 
                            max_new_tokens = 5, 
                            temperature = 0.001,
                        )
        self.pipe = pipe
    def predict (self, text):
        return self.pipe(generate_text(text))  