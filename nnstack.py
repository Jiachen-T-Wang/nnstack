import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim


### Define Neural Stack Models

class NNstack(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, prev_Val, prev_stg, dt, ut, vt):
    
    # prev_Val.shape = (t-1, m)
    # prev_stg.shape = (t-1, 1)
    # push signal dt is scalar in (0, 1)
    # pop signal ut is scalar in (0, 1)
    # vt.shape = (1, m)
    
    t_1, m = prev_Val.shape
    t = t_1 + 1
    
    # Update value matrix
    vt = vt.squeeze(0)
    Val = torch.cat((prev_Val, vt), dim=0)
    
    # Update strength vector
    stg = torch.zeros(t).to(device)
    stg[t-1] = dt
    
    for i in np.arange(t_1-1, -1, -1):
      temp = prev_stg[i] - max(0, ut)
      stg[i] = max(0, temp)
      ut = ut-stg[i]
      
    # Produce read vector rt.shape = (1, m)
    rt = torch.zeros(m).to(device)
    
    read = 1
    for i in np.arange(t-1, -1, -1):
      temp = max(0, read)
      coef = min(stg[i], temp)
      rt = rt + coef*Val[i]
      read = read - stg[i]
    
    return (Val, stg), rt
  
  
class Controller(nn.Module):
  def __init__(self, input_dim, hid_dim, n_layers, vocab_size=130):
    super().__init__()
    
    self.hid_dim = hid_dim
    
    self.vocab_size = vocab_size
    
    self.nnstack = NNstack()
    self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
    
    self.Wd = torch.zeros(hid_dim, requires_grad=True).to(device)
    self.Bd = torch.zeros(1, 1, requires_grad=True).to(device)
    
    self.Wu = torch.zeros(hid_dim, requires_grad=True).to(device)
    self.Bu = torch.zeros(1, 1, requires_grad=True).to(device)
    
    self.Wv = torch.zeros(hid_dim, vocab_size, requires_grad=True).to(device)
    self.Bv = torch.zeros(1, 1, vocab_size, requires_grad=True).to(device)
    
    self.Wo = torch.zeros(hid_dim, hid_dim, requires_grad=True).to(device)
    self.Bo = torch.zeros(1, 1, hid_dim, requires_grad=True).to(device)
    
  def forward(self, input, prev_State):
    
    if prev_State is None:
      prev_Val = torch.zeros(1, self.vocab_size).to(device)
      prev_stg = torch.zeros(1, 1).to(device)
      prev_hidden_cell = None
      prev_read = torch.zeros(self.vocab_size).to(device)
    else:
      (prev_Val, prev_stg), prev_hidden_cell, prev_read = prev_State
    
    
    input_aug = input + prev_read
    
    if prev_hidden_cell == None:
      output, hidden = self.rnn(input_aug)
    else:
      output, hidden = self.rnn(input_aug, prev_hidden_cell)
    
    # output.shape = [seq_len, 1, hidden_size]
    
    dt = torch.sigmoid(torch.matmul(output, self.Wd) + self.Bd)
    # dt.shape = (1, 1)
    
    ut = torch.sigmoid(torch.matmul(output, self.Wu) + self.Bu)
    # ut.shape = (1, 1)
    
    vt = torch.tanh(torch.matmul(output, self.Wv) + self.Bv)
    # vt.shape = (1, 1, vocab_size)
    
    ot = torch.tanh(torch.matmul(output, self.Wo) + self.Bo)
    # ot.shape = (1, 1, vocab_size)
    
    (Val, stg), rt = self.nnstack(prev_Val, prev_stg, dt, ut, vt)
    
    State = ((Val, stg), hidden, rt)
    
    return ot, State