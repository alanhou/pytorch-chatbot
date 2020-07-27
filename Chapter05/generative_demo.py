# -*- coding: utf-8 -*-
import torch, spacy, random, math, time
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

__author__ = 'Alan Hou'

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# python -m spacy download en
# python -m spacy download de
spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)][::-1]

SRC = Field(tokenize = tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)

TRG = Field(tokenize = tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields=(SRC,TRG))

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validtion examples: {len(valid_data.examples)}")
print(f"Number of test examples: {len(test_data.examples)}")

print(vars(train_data[0]))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f"Unique tokens in source(de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in targe(en) vocabulary: {len(TRG.vocab)}")

# 指定 GPU 或 CPU 进行训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置一些超参
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        # 给定参数
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    # 前向计算
    def forward(self, src):

        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        # 给定参数
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    # 前向计算
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(outputs.squeeze(0))

        return prediction, hidden, cell


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        # 给定参数
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, 'Hidden dimensions of encoder and decoder must be equal!'
        assert encoder.n_layers == decoder.n_layers, 'Encoder and decoder must have equal number of layers!'

    # 前向传播
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of encoder is used as tje initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> token
        input = trg[0, :]
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            # teacher_forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # 排序
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# 给定参数
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2seq(enc, dec, device).to(device)

# 初始化权重
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

PAD_INX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_INX)


# 训练过程
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/len(iterator)

# 验证
def evaluate(model, iterator, criterion):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)


# epoch time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - elapsed_mins*60)
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"Epoch: {epoch+1:02}|Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f}|Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\tVA1. Loss: {valid_loss:.3f}|Train PPL: {math.exp(valid_loss):7.3f}")


# prediction
model.load_state_dict(torch.load("tut1-model.pt"))
test_loss = evaluate(model, test_iterator, criterion)
print(f"| TEST Loss: {test_loss:.3f}|Train PPL: {math.exp(test_loss):7.3f} |")

