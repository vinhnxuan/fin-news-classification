import torch
from peft import AutoPeftModelForCausalLM
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

finetuned_model = "./trained_weigths/"
model_name = "NousResearch/Llama-2-7b-chat-hf"
compute_dtype = getattr(torch, "float16")
tokenizer = AutoTokenizer.from_pretrained(model_name,)

print(f"pytorch version {torch.__version__}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path = model_name,
#     device_map=device,
#     torch_dtype=compute_dtype,
#     quantization_config=bnb_config, 
#     low_cpu_mem_usage=True,
# )

# print(model)

# dsadadada

model = AutoPeftModelForCausalLM.from_pretrained(
     finetuned_model,
     torch_dtype=compute_dtype,
     return_dict=False,
     low_cpu_mem_usage=True,
     device_map=device,
)

merged_model = model.merge_and_unload()
# merged_model.save_pretrained("./merged_model",safe_serialization=True, max_shard_size="2GB")
# tokenizer.save_pretrained("./merged_model")

def evaluate(y_true, y_pred, filename):
    labels = ['positive', 'neutral', 'negative']
    mapping = {'positive': 2, 'neutral': 1, 'none':1, 'negative': 0}
    def map_func(x):
        return mapping.get(x, 1)
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report
    
    class_report = classification_report(y_true=y_true, y_pred=y_pred, output_dict= True)
    class_report_df = pd.DataFrame(class_report).transpose()
    print('\nClassification Report:')
    # Plot classification report
    plt.figure(figsize=(10, 6))
    sns.heatmap(class_report_df.iloc[:-1, :].T, annot=True, cmap='Blues', cbar=False)
    plt.title('Classification Report')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.savefig(filename + "_class_report.png")
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filename + "_confusion_matrix.png")
    print(conf_matrix)
    
X_train = pd.read_csv("scripts/x_train.csv")
X_eval = pd.read_csv("scripts/x_eval.csv")
X_test = pd.read_csv("scripts/x_test.csv")

import pickle
filehandler = open("scripts/y_true.pickle", 'rb') 
y_true = pickle.load(filehandler)
filehandler.close()
    
def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 10, 
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

y_pred = predict(X_test, merged_model, tokenizer)
evaluate(y_true, y_pred, filename="llama2_finetuned")

print("Unique labels in y_true:", np.unique(y_true))
print("Unique labels in y_pred:", np.unique(y_pred))