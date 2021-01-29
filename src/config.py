import pandas as pd
from transformers import BertModel, BertTokenizer

df = pd.read_csv("../input/train.csv")

BERT_MODEL = "bert-base-cased"
TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL)
LABEL_COLUMNS = df.columns[2:].to_list()
BATCH_SIZE=32
N_EPOCHS = 30