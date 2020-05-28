from model import AlignModel, Seq2Seq, EncoderLSTM, Decoder
from data_gen import dev_iter, vocab
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
attn = AlignModel()
enc = EncoderLSTM()
dec = Decoder()
special_ids = [vocab.stoi['<sos>'], vocab.stoi['<eos>'], vocab.stoi['<unk>'], vocab.stoi['<pad>']]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('./checkpoint/Seq2Seq_2020-05-28 16:21%.pth'))
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
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
        

def predict(model, criterion=criterion, test_set=dev_iter):
    model.eval()
    writer = SummaryWriter(comment='Seq2Seq')
    count = 0
    for batch in test_set:
        count+=1
        input_seq = batch.query.t()
        target_seq = batch.target.t()
        dummy_qry = input_seq
        dummy_trt = target_seq
        input_sample = input_seq[:10]
        target_sample = target_seq[:10]
        #打印示例
        print("The Query samples:\n")
        id2doc(input_sample, vocab)
        print("The Target samples:\n")
        id2doc(target_sample, vocab)
        output = model(input_seq, target_seq, teacher_forcing=False)
        output_sample = output.cpu().detach()[:10]
        output_sample = torch.max(output_sample,dim=2)[1]
        output_sample = output_sample.numpy()
        print("The Output samples:\n")
        id2doc(output_sample, vocab)
        output = output.permute(1,0,2)[1:].contiguous().view(-1,output.shape[-1])
        target_seq = target_seq.t()[1:].view(-1)
        loss = criterion(output, target_seq)
        print("Loss:%7.3f"%(loss.item()))
        writer.add_scalar('Test.', loss, count)
    writer.add_graph(model,[dummy_qry,dummy_trt])
    writer.close()
    return

    
if __name__ == '__main__':
    print('模型测试\n')
    predict(model)