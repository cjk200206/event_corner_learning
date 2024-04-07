import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, bidirectional=False):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.layers = []
        for i in range(self.num_layers):
            input_dim = self.input_channels if i == 0 else self.hidden_channels
            self.layers.append(ConvLSTMCell(input_dim, self.hidden_channels, self.kernel_size))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_tensor):
        b, _, h, w = input_tensor.size()
        hidden_states = []
        cell_states = []
        layer_output = input_tensor

        for i in range(self.num_layers):
            hidden, cell = self.layers[i].init_hidden(b, (h, w))
            hidden_states.append(hidden)
            cell_states.append(cell)

        for layer_idx in range(self.num_layers):
            output_inner = []
            hidden_states[layer_idx], cell_states[layer_idx] = self.layers[layer_idx](layer_output, (hidden_states[layer_idx], cell_states[layer_idx]))
            output_inner.append(hidden_states[layer_idx])
            layer_output = output_inner[-1]

        return output_inner[-1]

# Example usage:
batch_size = 16
seq_length = 10
input_channels = 3
hidden_channels = 64
kernel_size = (3, 3)
num_layers = 2

convlstm = ConvLSTM(input_channels, hidden_channels, kernel_size, num_layers)
input_tensor = torch.randn(batch_size, input_channels, 64, 64)  # Example input tensor
output_tensor = convlstm(input_tensor)
print("Output tensor shape:", output_tensor.shape)
