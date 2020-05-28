"""
the file contain some parameters
"""
import jieba
from tqdm import tqdm
MAX_LENGTH = 15
MIN_WORD_COUNT = 3
MAX_WORD_COUNT = 10
#some token that could fill the corpus
S = '<sos>'
E = '<eos>'
P = '<pad>'
U = '<unk>'


#params about file
PATH = r"./corpus/Gossiping-QA-Dataset.txt"
LOG_PATH = r"./checkpoint/"
SAVE_PATH = r"./corpus/params.pkl"
#params about training
embedding_size = 300
batch_size = 32
hidden_size = 300 #hidden layer of lstm
enc_seq_size = 20
dec_seq_size = 20
dropout = 0.8
epochs = 10
#function
