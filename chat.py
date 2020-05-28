from model import AlignModel, Seq2Seq, EncoderLSTM, Decoder
from data_gen import dev_iter, vocab
import jieba
import torch
import torch.nn as nn
attn = AlignModel()
enc = EncoderLSTM()
dec = Decoder()
special_ids = [vocab.stoi['<sos>'], vocab.stoi['<eos>'], vocab.stoi['<unk>'], vocab.stoi['<pad>']]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('./checkpoint/Seq2Seq_2020-05-28 16:21%.pth'))
model.eval()
#将输出转换为词汇,并打印
def id2doc(data, my_vocab, special_ids=special_ids):
    #data = data.numpy()
    res = []
    for text in data:
        tmp = []
        for num in text:
            if num not in special_ids:
                tmp.append(my_vocab.itos[num])
        res.append(tmp)
    for sen in res:
        print(sen)
        
def to_model(text, vocab=vocab):
    text = jieba.lcut(text)
    res = []
    for word in text:
        if word in vocab.stoi:
            res.append(vocab.stoi[word])
        else:
            res.append(vocab.stoi['<unk>'])
    return res, text

def chatbot(model):   
    while True:
        query = input()
        if query == '结束对话':
            break
        query, backup = to_model(query)
        query = torch.LongTensor(query).to(device)
        query = query.unsqueeze(0)
        output = model(query, query, teacher_forcing=False)
        output = output.cpu().detach()
        output = torch.max(output,dim=2)[1]
        output = output.numpy()
        print("The Query :\n")
        print(backup)
        print("The Answer :\n")
        id2doc(output, vocab)
    return
    
if __name__ == '__main__':
    print('尝试输入语句进行对话\n')
    chatbot(model)
