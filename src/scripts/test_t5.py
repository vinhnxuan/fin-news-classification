from libs.classifiers.t5 import T5Classifier
from libs.data_processor import DATA_PROCESSOR
from peft import LoraConfig

filename = "../data/all-data.csv"
MAXLENGTH = 128
data_processor = DATA_PROCESSOR(maxlen = MAXLENGTH, prompt_mode = False)
data_processor.read_csv(filename)

peft_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q", "v"], 
            lora_dropout=0.05, 
            bias="none", 
            task_type="SEQ_2_SEQ_LM"
        )

predictor = T5Classifier(maxlen = MAXLENGTH)
predictor.build_model()
#predictor.evaluate(data_processor, name = "llama2_org")
predictor.train(data_processor=data_processor, peft_config=peft_config)
#predictor.evaluate(data_processor, name = "llama2_finetuned_r32_alpha16_all")