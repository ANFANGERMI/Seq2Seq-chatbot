from torchtext.data import TabularDataset,Field,BucketIterator
from torchtext import vocab
import jieba
import re
import logging
import torch
from settings import batch_size, MAX_LENGTH

jieba.setLogLevel(logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize(text):
    regex = re.compile(r'\n')
    text = regex.sub('', text)
    return [word for word in jieba.cut(text) if word.strip()]


#定义torchtext使用的字段对象
FIELD = Field(sequential=True, use_vocab=True, tokenize=tokenize, fix_length=MAX_LENGTH, init_token = '<sos>',
            eos_token = '<eos>', lower = True)
datafields = [('query', FIELD), ('target', FIELD)]

#读取数据集
trn = TabularDataset(path='./train_q.csv', format='CSV', fields=datafields, skip_header=True)
tst = TabularDataset(path='./test_q.csv', format='CSV', fields=datafields, skip_header=True)
dev = TabularDataset(path='./dev_q.csv', format='CSV', fields=datafields, skip_header=True)

FIELD.build_vocab(trn)
VOCAB_SIZE = len(FIELD.vocab)
vocab = FIELD.vocab


train_iter = BucketIterator(trn, batch_size = batch_size, device=device, 
                            sort=False, sort_within_batch=False, repeat=False)
dev_iter = BucketIterator(dev, batch_size = batch_size, device=device,
                          sort=False, sort_within_batch=False, repeat=False)
tst_iter = BucketIterator(tst, batch_size = batch_size, device=device,
                          sort=False, sort_within_batch=False, repeat=False)



if __name__ == "__main__":
    print("该文件产生迭代器\n")
    print("词库大小为: %d\n" % VOCAB_SIZE)