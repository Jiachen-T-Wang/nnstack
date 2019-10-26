import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim



# Define Seq2Seq Models
class Encoder(nn.Module):
  def __init__(self, input_dim, hid_dim, n_layers, type='RNN'):
    super().__init__()
    
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    
    self.type = type
    
    self.embedding = nn.Embedding(input_dim, hid_dim)
    
    if type=='RNN':
      self.rnn = nn.RNN(input_dim, hid_dim, n_layers)
    elif type=='LSTM':
      self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
    elif type=='NNstack':
      self.rnn = Controller(input_dim, hid_dim, n_layers)
    
  def forward(self, src):
    
    #src.shape = (seq_len, vocab_size)
    
    seq_len, vocab_size = src.shape
    
    src = src.unsqueeze(1)
    
    #src.shape = (seq_len, 1, vocab_size)

    # if we use neural stack, both encoder and decoder use the same stack. 
    if self.type=='NNstack':
      input = src[0:1]
      hidden = None
      for i in range(1, seq_len):
        output, hidden = self.rnn(input, hidden)
        input = src[i:(i+1)]
    else:
      outputs, hidden = self.rnn(src)
    
    #hidden.shape = [n_layers, 1, hid_dim]
    #cell.shape = [n_layers, 1, hid_dim]
    
    return hidden
  
  
class Decoder(nn.Module):
  def __init__(self, output_dim, hid_dim, n_layers, type='RNN'):
    super().__init__()
    
    self.output_dim = output_dim
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    
    self.embedding = nn.Embedding(output_dim, hid_dim)
            
    if type=='RNN':
      self.rnn = nn.RNN(output_dim, hid_dim, n_layers)
    elif type=='LSTM':
      self.rnn = nn.LSTM(output_dim, hid_dim, n_layers)
    elif type=='NNstack':
      self.rnn = Controller(output_dim, hid_dim, n_layers)
        
    self.out = nn.Linear(hid_dim, output_dim)
    
  def forward(self, input, hidden):
    #input.shape = [1, vocab_len]
    
    #hidden.shape = [n_layers, 1, hid_dim]
    #cell.shape = [n_layers, 1, hid_dim]
    
    input = input.unsqueeze(1)
    
    #input.shape = [1, 1, vocab_len]
    
    # embedded = self.embedding(input)
        
    output, hidden = self.rnn(input, hidden)
    
    #output.shape = (1, 1, hid_dim)
    
    prediction = self.out(output.squeeze(1))
    
    return prediction, hidden
  
  
class Seq2seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
    
    if (encoder.hid_dim!=decoder.hid_dim) or (encoder.n_layers!=decoder.n_layers):
      sys.exit('ENCODER AND DECODER MUST HAVE SAME DIM!')
      
  def forward(self, src, tgt, teacher_force = 0.75):
    
    # BATCH SIZE = 1
    # src = [seq_len, vocab_size]
    # tgt = [seq_len, vocab_size]
    
    seq_len, vocab_size = tgt.shape[0], tgt.shape[1]
    
    # tgt_vocab_size = self.decoder.output_dim
    
    # tensor to store outputs
    outputs = torch.zeros(seq_len, vocab_size).to(self.device)
    
    #last hidden state of the encoder is used as the initial hidden state of the decoder
    hidden = self.encoder(src)
    
    #first input to the decoder is the <sos> tokens
    input = tgt[0:1]
    
    for t in range(1, seq_len):
      
      #insert input token embedding, previous hidden and previous cell states
      #receive output tensor (predictions) and new hidden and cell states
      output, hidden = self.decoder(input, hidden)
            
      #place predictions in a tensor holding predictions for each token
      outputs[t] = output
      
      #get the highest predicted token from our predictions
      top1 = output.argmax(1)
      
      # IN EVAL PHASE, stop training / prediction if top1 is EOS
      if self.training == False:
        if top1==129: break
      
      #transform top1 to one-hot encoding
      top1_val = top1.data.tolist()[0]
      top1 = [0 for _ in range(130)]
      top1[top1_val] = 1
      
      top1 = torch.Tensor(top1).to(self.device)
      top1 = top1.unsqueeze(0)
      
      tf = np.random.choice([1, 0], p=[teacher_force, 1-teacher_force])
      
      if tf:
        input = tgt[t:(t+1)]
      else:
        input = top1
        
    return outputs


### Define Training & Evaluation Process

def train(model, train_set, optimizer, criterion):
  model.train()
  
  epoch_loss = 0
  
  train_src, train_tgt = train_set['src'], train_set['tgt']
  
  BATCH_SIZE = len(train_src)
    
  for i in range(BATCH_SIZE):
    
    src = torch.Tensor(train_src[i]).to(model.device)
    tgt = torch.Tensor(train_tgt[i]).to(model.device)
        
    optimizer.zero_grad()
        
    output = model(src, tgt)
        
    #tgt.shape = [seq_len, vocab_size]
    #output.shape = [seq_len, vocab_size]
    
    output = output[1:]
    tgt = tgt[1:]
    
    #trg.shape = [seq_len-1, vocab_size]
    #output.shape = [seq_len-1, vocab_size]
    
    loss = criterion(output, tgt.argmax(dim=1))
    loss.backward()
                
    optimizer.step()
        
    epoch_loss += loss.item()

  return epoch_loss / BATCH_SIZE


def evaluate(model, test_set, criterion):
  model.eval()
  
  epoch_loss = 0
  coarse = 0
  fine_lst = []
  
  test_src = test_set['src']
  test_tgt = test_set['tgt']
  
  BATCH_SIZE = len(test_src)
    
  with torch.no_grad():
    
    for i in range(BATCH_SIZE):
      
      src = torch.Tensor(test_src[i]).to(model.device)
      tgt = torch.Tensor(test_tgt[i]).to(model.device)

      output = model(src, tgt, 0) #turn off teacher forcing
      
      output = output[1:]
      tgt = tgt[1:]
      
      output = output.data.tolist()
      tgt = tgt.data.tolist()
      
      fine_i = 0
      
      output_len = len(output)
      seq_len = len(tgt)
      
      for i in range(min(seq_len, output_len)):
        output_i = np.argmax(output[i])
        tgt_i = np.argmax(tgt[i])
        if output_i == tgt_i: fine_i += 1
        else: break
      
      fine_lst.append(fine_i/seq_len)
      if fine_i == seq_len: coarse += 1
        
  return coarse / BATCH_SIZE, np.mean(fine_lst)