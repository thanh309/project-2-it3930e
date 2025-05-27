import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class BayesianDense(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_sigma = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w_mu, mode='fan_in')
        nn.init.constant_(self.w_sigma, -3)
        nn.init.constant_(self.b_mu, 0.1)
        nn.init.constant_(self.b_sigma, -3)

    def forward(self, x):
        w = Normal(self.w_mu, torch.exp(self.w_sigma)).rsample()
        b = Normal(self.b_mu, torch.exp(self.b_sigma)).rsample()
        return F.linear(x, w, b)

class MyModel(nn.Module):
    def __init__(self, input_features=10, timesteps=5, num_classes=5):
        super().__init__()
        assert input_features == 10, 'only 10 features supported right now'

        self.input_dense = BayesianDense(input_features, 32)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)

        self.bayesian_layers = nn.ModuleList([
            BayesianDense(64, 64),
            BayesianDense(64, 48),
            BayesianDense(48, 48),
            BayesianDense(48, 32)
        ])

        self.dropout = nn.Dropout(0.1)
        self.final_dense = BayesianDense(32, num_classes)

        self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        batch_size, T, _ = x.shape

        x = self.swish(self.input_dense(x))  # (batch, T, 32)
        lstm_out, _ = self.lstm(x)  # (batch, T, 64)

        x = lstm_out
        for layer in self.bayesian_layers:
            x = self.swish(layer(x))

        x = self.dropout(x)
        logits = self.final_dense(x)  # (batch, T, num_classes)
        return logits
