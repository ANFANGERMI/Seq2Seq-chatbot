import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import hidden_size,batch_size,embedding_size,dropout,MAX_LENGTH
from data_gen import vocab
#first define the encoder, a BiLSTM
vocab_size = len(vocab)
class EncoderLSTM(nn.Module):
    def __init__(self, batch_size=batch_size, embedding_size=embedding_size,
                 hidden_size=hidden_size, n_layers=1, dropout=dropout):
        """

        :param batch_size:the batch size
        :param seq_size: length of input seq
        :param embedding_size: the dimension of word vector
        :param hidden_size: the num of hidden unit
        :param n_layers: the num of hidden layer
        :param dropout: prob of dropout
        """
        super(EncoderLSTM, self).__init__()
        #批训练的训练集大小
        self.batch_size = batch_size
        #词向量的维度
        self.embedding_size = embedding_size
        #隐藏层参数，原文中使用了1000个，此处改为500个
        self.hidden_size = hidden_size
        #输出单词总数
        self.num_class = vocab_size
        #隐藏层层数，默认为一层
        self.n_layers = n_layers
        #原文中使用了gradient cliping
        self.embed = nn.Embedding(self.num_class, self.embedding_size).cuda()
        self.embed_dropout = nn.Dropout(dropout).cuda()
        #initialize the lstm unit
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.n_layers,
                            batch_first=True,
                            bidirectional=False).cuda()
        #将最后状态的叠加经过Linear加上activation传递给decoder作为0状态
        #self.output_to_dec  = nn.Linear(self.hidden_size, self.hidden_size)
        #self.cell = torch.zeros(self.n_layers,batch_size,hidden_size).cuda()


    def forward(self, input_seq):
        """

        :param input_seq:
        :return:
        """
        #此处得到输出的context vector以及一个包含了hiddenstate以及cell的tuple
        input_embed = self.embed(input_seq)
        input_embed = self.embed_dropout(input_embed)
        #annotation shape [batch,seq_len,hidden_size]
        annotation, hiddens = self.lstm(input_embed)
        #unpack padding
        #outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        #将输出的隐状态拼接成为annotations
        #hiddens是一个tuple,包含了最终的hiddenstate和cell
        hidden = hiddens[0]
        cell = hiddens[1]
        init_state = tuple([hidden,cell])
        return annotation, init_state


#此处定义原文中的对齐模型，与Luong Attention不同，该文章中每一个输出位置对应输入序列中的分数由另一个对齐模型训练得到
#W,U,V分别对应decoder前一个hiddenstate的matrix,encoder的annotations的matrix
#原文提到使用maxout,此处不使用maxout
class AlignModel(nn.Module):
    def __init__(self, hidden_size = hidden_size):
        super(AlignModel, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, annotation, dec_hidden_state):
        batch_size = annotation.shape[0]
        seq_len = annotation.shape[1]
        #print(dec_hidden_state.shape)
        #print(annotation.shape)
        #让隐藏状态第二个维度与encoder相同
        #dec_hidden_state: [batch, 1, dec_hid_dim]
        dec_hidden_state = dec_hidden_state.permute(1,0,2)
        dec_hidden_state = dec_hidden_state.repeat(1, seq_len, 1)
        #[batch_size,seq_len,hidden_size]
        energy = F.tanh(self.attn(torch.cat((dec_hidden_state, annotation), dim=2)))
        #energy相当于encoder每个位置对decoder每一个位置的评分
        energy = energy.permute(0, 2, 1)
        #v:[batch_size,1,dec_hid_dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        #[b,1,dec] x [b,dec,seq_len]
        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, num_class=vocab_size, batch_size=batch_size,
                embedding_size=embedding_size,hidden_size=hidden_size,
                 n_layers=1,dropout = dropout):
        super().__init__()
        self.attention = AlignModel()
        # 批训练的训练集大小
        self.batch_size = batch_size
        # 词向量的维度
        self.embedding_size = embedding_size
        # 隐藏层参数，原文中使用了1000个，此处改为500个
        self.hidden_size = hidden_size
        # 隐藏层层数，默认为一层
        self.n_layers = n_layers
        # 原文中使用了gradient cliping
        self.num_class = vocab_size
        self.dropout = dropout
        # initialize the lstm unit
        self.embed = nn.Embedding(self.num_class, self.embedding_size).cuda()
        self.embed_dropout = nn.Dropout(self.dropout).cuda()
        self.lstm = nn.LSTM(input_size=self.embedding_size+self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.n_layers,
                            batch_first=True,
                            bidirectional=False)
        self.out = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, init_state, annotation, answer_seq):
        """

        :param hidden:
        :param annotation:
        :param output_seq:shape [batch,seq_len]
        :return:
        """
        #这一步计算annotation向量与前一个隐状态的attention
        #a = [batch_size,1,seq_len]
        embed_target = self.embed(answer_seq)
        #[seq_len,batch_size,embed]
        answer_seq = self.embed_dropout(embed_target)
        init_hidden = init_state[0]
        a = self.attention(annotation, init_hidden)
        #[batch,1,seq_len]
        a = a.unsqueeze(1)
        #将原文中decoder每一步的输入看作上一步的输出加一个加权向量C
        #[1,seq] x [seq,hid]
        weighted = torch.bmm(a, annotation)
        #print(weighted.shape)
        #print(answer_seq.shape)
        #input shape:[batch,1,hid + embbed]
        lstm_input = torch.cat((answer_seq, weighted), dim=2)
        #提供enc最后的隐状态作为初始的dec隐状态，并输入input
        output, hiddens = self.lstm(lstm_input, init_state)
        #此处在decoder隐藏层上增加线性层，输出转换为score
        output = self.out(output)
        return output, hiddens


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.seq_len = MAX_LENGTH
        self.vocab_size = vocab_size

    def forward(self, query_seq, answer_seq, teacher_forcing=True):
        """

        :param input_seq: the query with length 20
        :param output_seq: the answer with size 20
        :return:
        """
        batch_size = query_seq.shape[0]
        #最终的decoder的输出,基于词典大小的概率向量
        pred_seq = torch.zeros(batch_size, self.seq_len, self.vocab_size).to(self.device)
        #输入形状:[batch_size, seq_len, embedding_size]
        encoder_output, init_state = self.encoder(query_seq)
        #[batch,seq]
        #answer_seq = answer_seq.permute(1, 0)
        output = answer_seq[:,0].unsqueeze(1)
        state = init_state
        for t in range(self.seq_len):
            #在每一个decoder的输入中插入前一个隐藏状态
            if t == 0:
                #the first time step
                output, state = self.decoder(state, encoder_output, output)
            else:
                #else step
                output, state = self.decoder(state, encoder_output, output)
            pred_seq[:,t,:] = output.squeeze(1)
            top = output.max(2)[1]
            #直接使用teacher forcing进行训练
            if teacher_forcing:
                output = answer_seq[:, t].unsqueeze(1)
            #测试的时候不用teacher_forcing
            else:
                output = top
        return pred_seq

