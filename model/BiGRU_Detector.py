
import torch
import torch.nn as nn

class BiGRU_Detector(nn.Module):

    def __init__(self, input_size, hidden_size, num_layer=1):
        super(BiGRU_Detector, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layer,\
                          bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, 1)

    def forward(self, input):
        rnn_output, _  = self.rnn(input)
        output = nn.Sigmoid()(self.linear(rnn_output))
        return output


if __name__ == '__main__':
    model = BiGRU_Detector(100, 50, 1)
    input = torch.Tensor(5, 128, 100)
    output = model(input)
    print()