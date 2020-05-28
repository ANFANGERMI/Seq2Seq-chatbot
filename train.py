from model import AlignModel, Seq2Seq, EncoderLSTM, Decoder
from settings import batch_size, epochs, LOG_PATH
from data_gen import train_iter, dev_iter, vocab
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
import os
attn = AlignModel()
enc = EncoderLSTM()
dec = Decoder()
special_ids = [vocab.stoi['<sos>'], vocab.stoi['<eos>'], vocab.stoi['<unk>'], vocab.stoi['<pad>']]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)
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

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            

def count_parammeters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("模型有%d可训练参数"  %count_parammeters(model))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])


def train(model, train_iter, optimizer=optimizer, criterion=criterion, clip=1):
    model.train()
    epoch_loss = 0
    count = 0
    samples = 0
    dummy_qry = None
    dummy_trt = None
    #writer = SummaryWriter(comment='Seq2Seq-Chatbot')
    for batch in train_iter:
        count+=1
        #[batch_size,seq_len]
        input_seq = batch.query.t()
        target_seq = batch.target.t()
        samples += input_seq.shape[0]
        dummy_qry = input_seq
        dummy_trt = target_seq
        input_sample = input_seq[:5]
        target_sample = target_seq[:5]
        #打印示例
        print("The Query samples:\n")
        id2doc(input_sample, vocab)
        print("The Target samples:\n")
        id2doc(target_sample, vocab)
        optimizer.zero_grad()
        output = model(input_seq, target_seq,torch.device('cuda'))
        output_sample = output.cpu().detach()[:5]
        output_sample = torch.max(output_sample,dim=2)[1]
        output_sample = output_sample.numpy()
        #output = torch.max(output,dim=2)[1]
        print("The Output samples:\n")
        id2doc(output_sample, vocab)
        output = output.permute(1,0,2)[1:].contiguous().view(-1,output.shape[-1])
        target_seq = target_seq.t()[1:].view(-1)
        loss = criterion(output, target_seq)
        print("Loss:%7.3f"%(loss.item()))
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        #writer.add_scalar('Train.', epoch_loss, count)
    
    #writer.add_graph(model, [dummy_qry, dummy_trt])
    #writer.close()
    
    return epoch_loss / samples

def evaluate(model, valid_set, valid_target,label, criterion, batch_size=batch_size):
    model.eval()
    epoch_loss = 0
    valid_batch_index = make_batch(valid_set)

    with torch.no_grad():
        for index in valid_batch_index:
            input_seq = valid_set[index:index+batch_size]
            target_seq = valid_target[index:index+batch_size]
            output = model(input_seq, target_seq)
            output = output.view(-1, output.shape[-1])
            #展平标签
            label = label.view(-1)
            loss = criterion(output, target_seq)
            epoch_loss += loss.item()
    return epoch_loss / batch_size


#记录时间开销
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



#best_valid_loss = float('inf')
if __name__ == '__main__':
    print("开始训练:\n")
    init_weights(model)
    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion)
        #valid_loss = evaluate(model, val_var, output_val_var, val_label,criterion)
        end_time = time.time()
        epoch_mins,epoch_secs = epoch_time(start_time, end_time)
        #best_valid_loss = valid_loss
        #state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        #torch.save(state, 'seq2seq_model.pth')
        print(f'Epoch: {epoch + 1:2} | Time: {epoch_mins}m {epoch_secs}s')
        print('\tTrain Loss:%7.3f  | Perplexity:%7.3f '%(train_loss, math.exp(train_loss)))
        save_name = 'Seq2Seq_{}%.pth'.format(time.strftime("%Y-%m-%d %H:%M", time.localtime()))
        #state = {'model': model.state_dict()}
        torch.save(model.state_dict(), os.path.join(LOG_PATH,save_name))
        #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
