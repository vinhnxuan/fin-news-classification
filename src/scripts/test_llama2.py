from libs.classifiers.llama2 import LLAMA2Classifier
from libs.data_processor import DATA_PROCESSOR
from peft import LoraConfig

filename = "../data/all-data.csv"
MAXLENGTH = 128
data_processor = DATA_PROCESSOR(maxlen = MAXLENGTH, prompt_mode = True)
data_processor.read_csv(filename)

peft_config = LoraConfig(
                lora_alpha=16, 
                lora_dropout=0.1,
                r=32,
                bias="none",
                #target_modules="all-linear",
                target_modules = [
                        "q_proj",
                        #"up_proj",
                        "o_proj",
                        "k_proj",
                        #"down_proj",
                        #"gate_proj",
                        "v_proj"
                        ],
                task_type="CAUSAL_LM",
        )

predictor = LLAMA2Classifier()
predictor.build_model()
#predictor.evaluate(data_processor, name = "llama2_org")
predictor.train(data_processor=data_processor, peft_config=peft_config)
predictor.evaluate(data_processor, name = "t5_finetuned")