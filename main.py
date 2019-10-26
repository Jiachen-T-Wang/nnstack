import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from nnstack import *
from encoder_decoder import *


## Prepare Training Set

def gen_pair(lo, hi):
  k = np.random.choice(np.arange(lo, hi+1))
  src = ''
  for _ in range(k):
    i = np.random.choice(range(97, 122))    # TODO: back to 0, 128
    src = src + chr(i)
  tgt = src[::-1]
  return src, tgt


def one_hot_encode(data):
  integer_encoded = [ord(token) for token in data]
  
  onehot_encoded = list()
  
  # add SOS
  sos = [0 for _ in range(130)]
  sos[128] = 1
  onehot_encoded.append(sos)
  
  for value in integer_encoded:
    letter = [0 for _ in range(130)]
    letter[value] = 1
    onehot_encoded.append(letter)

  # add EOS
  eos = [0 for _ in range(130)]
  eos[129] = 1
  onehot_encoded.append(eos)
  
  return np.array(onehot_encoded)


def make_set(lo, hi, n):
  train_set = {'src': [], 'tgt': []}
  for _ in range(n):
    src, tgt = gen_pair(lo, hi)
    src_encode = one_hot_encode(src)
    tgt_encode = one_hot_encode(tgt)
    train_set['src'].append(src_encode)
    train_set['tgt'].append(tgt_encode)

  train_set['src'] = np.array(train_set['src'])
  train_set['tgt'] = np.array(train_set['tgt'])
  
  return train_set


# Phase1: 8 16
# Phase2: 16 32
# Phase3: 32 64
# Test: 65 128
train_set1 = make_set(8, 16, 1000)
train_set2 = make_set(8, 32, 1000)
train_set3 = make_set(8, 64, 1000)
test_set = make_set(65, 128, 100)


## Build the model for RNN
INPUT_DIM = 130
OUTPUT_DIM = 130

HID_DIM = 256
N_LAYERS = 2

TYPE = 'RNN' # change the model type for LSTM and Neural Stack architecture. 

encoder = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, TYPE)
decoder = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, TYPE)

model = Seq2seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

# loss function
criterion = nn.CrossEntropyLoss()

model.apply(init_weights)


N_EPOCHS = 20

curriculum = [3, 6]

train_loss_lst = []
valid_loss_lst = []

train_set = make_set(8, 64, 10)
test_set = make_set(8, 64, 20)

for epoch in range(N_EPOCHS):
  
  """
  ## I abandoned curriculum learning since it's not helpful. 
  if epoch < curriculum[0]:
    train_set = train_set1
  elif epoch < curriculum[1]:
    train_set = train_set2
  else:
    train_set = train_set3
  """
    
  train_loss = train(model, train_set, optimizer, criterion)
  valid_loss = evaluate(model, test_set, criterion)
  
  train_loss_lst.append(train_loss)
  valid_loss_lst.append(valid_loss)
  
  print('EPOCH', epoch, ':')
  print('  TRAIN_LOSS:', train_loss)
  print('  VALID_LOSS:', valid_loss)