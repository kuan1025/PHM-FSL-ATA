import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.backbone import Linear_fw, Conv2d_fw, BatchNorm2d_fw, BatchNorm1d_fw

# ---- device dtypes ----
if torch.cuda.is_available():
  dtype = torch.cuda.FloatTensor
  dtype_l = torch.cuda.LongTensor
else:
  dtype = torch.FloatTensor
  dtype_l = torch.LongTensor  # <- fix

def gmul(input):
  W, x = input
  # x: (bs, N, num_features)
  # W: (bs, N, N, J)
  W_size = W.size()
  N = W_size[-2]
  W = W.split(1, 3)
  W = torch.cat(W, 1).squeeze(3)   # (bs, J*N, N)
  output = torch.bmm(W, x)         # (bs, J*N, num_features)
  output = output.split(N, 1)
  output = torch.cat(output, 2)    # (bs, N, J*num_features)
  return output

class Gconv(nn.Module):
  FWT = False
  def __init__(self, nf_input, nf_output, J, bn_bool=True):
    super(Gconv, self).__init__()
    self.J = J
    self.num_inputs = J * nf_input         
    self.num_outputs = nf_output
    self.fc = nn.Linear(self.num_inputs, self.num_outputs) if not self.FWT else Linear_fw(self.num_inputs, self.num_outputs)

    self.bn_bool = bn_bool
    if self.bn_bool:
      self.bn = nn.BatchNorm1d(self.num_outputs, track_running_stats=False) if not self.FWT else BatchNorm1d_fw(self.num_outputs, track_running_stats=False)

  def forward(self, input):
    W = input[0]

    x = gmul(input)  # (bs, N, J*num_features)

    # ---- hot-fix: self.num_inputs ----
    need = self.num_inputs - x.size(-1)
    if need > 0:

      pad = torch.zeros(x.size(0), x.size(1), need, device=x.device, dtype=x.dtype)
      x = torch.cat([x, pad], dim=-1)
    elif need < 0:

      x = x[..., :self.num_inputs]
    # -----------------------------------------------------------------

    x_size = x.size()               # (bs, N, self.num_inputs)
    x = x.contiguous().view(-1, self.num_inputs)  # (bs*N, self.num_inputs)
    x = self.fc(x)                  # (bs*N, num_outputs)

    if self.bn_bool:
      x = self.bn(x)
    x = x.view(*x_size[:-1], self.num_outputs)  # (bs, N, num_outputs)
    return W, x

class Wcompute(nn.Module):
  FWT = False
  def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
    super(Wcompute, self).__init__()
    self.num_features = nf
    self.operator = operator
    self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1) if not self.FWT else Conv2d_fw(input_features, int(nf * ratio[0]), 1, stride=1)
    self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]), track_running_stats=False) if not self.FWT else BatchNorm2d_fw(int(nf * ratio[0]), track_running_stats=False)
    self.drop = drop
    if self.drop:
      self.dropout = nn.Dropout(0.3)
    self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1) if not self.FWT else Conv2d_fw(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
    self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]), track_running_stats=False) if not self.FWT else BatchNorm2d_fw(int(nf * ratio[1]), track_running_stats=False)
    self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1) if not self.FWT else Conv2d_fw(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
    self.bn_3 = nn.BatchNorm2d(nf*ratio[2], track_running_stats=False) if not self.FWT else BatchNorm2d_fw(nf*ratio[2], track_running_stats=False)
    self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1) if not self.FWT else Conv2d_fw(nf*ratio[2], nf*ratio[3], 1, stride=1)
    self.bn_4 = nn.BatchNorm2d(nf*ratio[3], track_running_stats=False) if not self.FWT else BatchNorm2d_fw(nf*ratio[3], track_running_stats=False)
    self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1) if not self.FWT else Conv2d_fw(nf, num_operators, 1, stride=1)
    self.activation = activation

  def forward(self, x, W_id):
    # x: (bs, N, num_features)
    W1 = x.unsqueeze(2)
    W2 = torch.transpose(W1, 1, 2)      # (bs, N, N, num_features)
    W_new = torch.abs(W1 - W2)          # (bs, N, N, num_features)
    W_new = torch.transpose(W_new, 1, 3)  # (bs, num_features, N, N)

    # ---- hot-fix:（fix 132 vs 133）----
    C_expected = self.conv2d_1.weight.size(1)
    C_in = W_new.size(1)
    if C_in == C_expected - 1:
      pad = torch.ones(W_new.size(0), 1, W_new.size(2), W_new.size(3),
                       device=W_new.device, dtype=W_new.dtype)
      W_new = torch.cat([W_new, pad], dim=1)
    elif C_in > C_expected:
      W_new = W_new[:, :C_expected, ...]
    # -------------------------------------------------------------------

    W_new = self.conv2d_1(W_new)
    W_new = self.bn_1(W_new)
    W_new = F.leaky_relu(W_new)
    if self.drop:
      W_new = self.dropout(W_new)

    W_new = self.conv2d_2(W_new)
    W_new = self.bn_2(W_new)
    W_new = F.leaky_relu(W_new)

    W_new = self.conv2d_3(W_new)
    W_new = self.bn_3(W_new)
    W_new = F.leaky_relu(W_new)

    W_new = self.conv2d_4(W_new)
    W_new = self.bn_4(W_new)
    W_new = F.leaky_relu(W_new)

    W_new = self.conv2d_last(W_new)
    W_new = torch.transpose(W_new, 1, 3)  # (bs, N, N, num_operators)

    if self.activation == 'softmax':
      #  softmax
      W_new = W_new - W_id.expand_as(W_new) * 1e8
      W_new = torch.transpose(W_new, 2, 3).contiguous()   # (bs, N, num_ops, N)
      W_new_size = W_new.size()
      W_new = W_new.view(-1, W_new.size(3))
      W_new = F.softmax(W_new, dim=1)
      W_new = W_new.view(W_new_size)
      W_new = torch.transpose(W_new, 2, 3)

    elif self.activation == 'sigmoid':
      W_new = torch.sigmoid(W_new)
      W_new *= (1 - W_id)
    elif self.activation == 'none':
      W_new *= (1 - W_id)
    else:
      raise NotImplementedError

    if self.operator == 'laplace':
      W_new = W_id - W_new
    elif self.operator == 'J2':
      # J2: concat [I, W]，J=2
      W_new = torch.cat([W_id, W_new], 3)
    else:
      raise NotImplementedError

    return W_new  # (bs, N, N, J)

class GNN_nl(nn.Module):
  def __init__(self, input_features, nf, train_N_way):
    super(GNN_nl, self).__init__()
    self.input_features = input_features
    self.nf = nf
    self.num_layers = 2

    for i in range(self.num_layers):
      if i == 0:
        module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        module_l = Gconv(self.input_features, int(nf / 2), 2)
      else:
        module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
      self.add_module('layer_w{}'.format(i), module_w)
      self.add_module('layer_l{}'.format(i), module_l)

    self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
    self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, train_N_way, 2, bn_bool=False)

  def forward(self, x):
    # x: (bs, N, input_features)
    W_init = torch.eye(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)  # (bs, N, N, 1)

    for i in range(self.num_layers):
      Wi = self._modules['layer_w{}'.format(i)](x, W_init)    # (bs, N, N, J)
      x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
      x = torch.cat([x, x_new], 2)                            

    Wl = self.w_comp_last(x, W_init)
    out = self.layer_last([Wl, x])[1]
    return out
