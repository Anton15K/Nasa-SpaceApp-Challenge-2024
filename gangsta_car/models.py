import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNETEventClassifier(nn.Module): #Outputs whether there is an event in the waveform or not
    def __init__(self, input_channels=1, sequence_length=600, num_classes=1):
        super(ConvNETEventClassifier
    , self).__init__()

        #Convolutional blocks
        self.conv_layers = nn.Sequential(
            self.repeated_block(in_channels=input_channels), #1

            self.repeated_block(), #2
            self.repeated_block(), #3
            self.repeated_block(), #4
            self.repeated_block(), #5
            self.repeated_block(), #6
            self.repeated_block(), #7
            self.repeated_block(), #8

        )

        final_seq_length = sequence_length // (2**8)

        # Fully connected layer
        self.fc = nn.Linear(32 * final_seq_length, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = torch.sigmoid(x)
        return x

    def repeated_block(self, in_channels = 32, out_channels = 32, kernel_size=3, padding=1, pool_kernel_size=2):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)
        )

class BiLSTMEventDetector(nn.Module):
    def __init__(self, input_channels=1, hidden_size=64, num_layers=2, dropout_prob=0.6, sequence_length=600):
        super(BiLSTMEventDetector, self).__init__()

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(p=dropout_prob)

        #Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, 1)

        #Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        # Apply fully connected layer to each time step
        fc_out = self.fc(lstm_out)
        # Sigmoid
        out = self.sigmoid(fc_out)
        return out  # Shape: [batch_size, sequence_length, 1]
