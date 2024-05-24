import torch
import torchaudio
from torch.nn import functional as F
from torch import nn
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class lstm(nn.Module):
    def __init__(self, input_size=80, hidden_size=64, num_layers=2, num_classes=10,in_channels=1):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            #torch.nn.ReLU(),
            )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.seq(x)
        _n, _c, _h, _w = x.shape
        #print(x.shape)
        _x = x.permute(0, 3, 2, 1)
        _x = _x.reshape(_n, _w, _h * _c)
        
        h0 = torch.zeros(2 * 1, _n, 64).cuda()
        c0 = torch.zeros(2 * 1, _n, 64).cuda()
        
        hsn, (hn, cn) = self.lstm(_x, (h0, c0))
        
        out = self.fc(hsn[:, -1, :])
        return out

