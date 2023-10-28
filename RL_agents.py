import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

""" RL- Agent / LSTM+Linear nn
        params:
            -
    """
class PNG_LSTM(nn.Module):
    def __init__(self, input_size, n_actions, hidden_dim=12, num_layers=1):
        super(PNG_LSTM, self).__init__()

        # self.net = nn.Sequential(
        #     nn.Linear(input_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, n_actions)
        # )

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_actions  = n_actions

        # batch_size=True - is a very important flag for dimensions
        self.lstm   = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # self.rnn    = nn.RNN( input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, n_actions)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)

        """for simple RNN"""
        # out, (hn, cn) = self.lstm(x)
        # out, h = self.lstm(x)
        """for traditional LSTM"""
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # out = out.contiguous.view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, (hn, cn)

def get_action(net, start_state, range_values):
    start_state_t = torch.FloatTensor(start_state)
    logits_a, (_, _) = net(start_state_t)
    probs_a = F.softmax(logits_a, dim=2).detach().numpy() # detach -> new tnsor without gradients

    acts = []
    range_acts = np.arange(range_values.shape[0])
    for i in np.arange(probs_a.shape[1]):
        acts.append(np.random.choice(range_acts, p=probs_a[0, i]))

    values = range_values[acts]
    return acts, list(values)

def distributor_acts(option_args = []):
    seed = option_args.pop(-1)
    np.random.seed(seed)
    acts, vals = get_action(*option_args)
    return acts, vals
