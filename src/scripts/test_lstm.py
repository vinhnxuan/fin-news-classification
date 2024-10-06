from libs.classifiers.lstm import LSTMClassifer 
from libs.data_processor import DATA_PROCESSOR

filename = "../data/all-data.csv"
MAXLENGTH = 128
data_processor = DATA_PROCESSOR(maxlen = MAXLENGTH, tokenize_mode = True)
data_processor.read_csv(filename)

predictor = LSTMClassifer()
predictor.build_model(max_len = MAXLENGTH)
predictor.train(data_processor=data_processor)
predictor.evaluate(data_processor)