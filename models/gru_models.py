import torch
import torch.nn as nn


class KeypointsGRU(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_layers=1,
        hidden_dim=64,
        sequence_length=60,
        bidirectional=True,
        device = None
    ):
        super(KeypointsGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_units = hidden_dim
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.device = device
        
        self.gru = nn.GRU(
            self.num_features,
            self.hidden_units,
            self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )
   
    def forward(self, x):
        '''
        input shape [N, L, num_features]
        '''
        batch_size = x.shape[0]
        hidden_dim = 2*self.num_layers
        
        h0 = torch.zeros(hidden_dim, batch_size, self.hidden_units).to(self.device)
        x, hn = self.gru(x, h0)
        x = x[:, -1]
        
        return self.output_layers(x)  
    