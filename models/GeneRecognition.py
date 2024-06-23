import torch
import torch.nn as nn
from enum import Enum


class ModelType(Enum):
    LSTM = 1
    GRU = 2


# Define a simple model (e.g., using mean pooling)
class GeneRecognitionLSTM(nn.Module):
    def __init__(self, vocabulary, embedding_dimension, model_type=ModelType.LSTM, embedding=None):
        super(GeneRecognitionLSTM, self).__init__()

        self.vocabulary = vocabulary
        hidden_size = embedding_dimension * 4
        self.embedding = embedding or nn.Embedding(self.vocabulary.__len__(), embedding_dimension,
                                      padding_idx=self.vocabulary["pad"])
        rnn_layer = torch.nn.LSTM if model_type == ModelType.LSTM else torch.nn.GRU
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.rnn = rnn_layer(embedding_dimension, hidden_size, batch_first=True)
        # self.linear1 = torch.nn.Linear(self.hidden_size, self.hidden_size*2)
        self.linear = torch.nn.Linear(hidden_size, 1)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        # self.activation = torch.nn.Sigmoid()

    def forward(self, input_tensor, lengths):

        # embed on the device
        embedded = self.embedding(input_tensor)

        # # pack on the cpu, consider doing this so lstm ignores paddings
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        # packed = embedded
        lstm_output, _ = self.rnn(packed)

        # unpacked = lstm_output

        # # unpack on the cpu, consider doing this so lstm ignores paddings
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True,
                                                             padding_value=self.vocabulary["pad"])
        x = self.relu(unpacked[torch.arange(unpacked.size(0)), [length -1 for length in lengths]])
        x = self.linear1(self.norm(x))
        return self.linear(self.dropout(x))
        # output = self.linear1(self.dropout(self.relu(final_hidden_state)))
        # return self.linear(self.dropout(self.relu(output)))


class SplitLSTMModel(nn.Module):
    def __init__(self, vocabulary, embedding_dimension, model_type=ModelType.LSTM):
        super(SplitLSTMModel, self).__init__()
        self.vocabulary = vocabulary
        self.hidden_size = 64
        self.embedding = nn.Embedding(self.vocabulary.__len__(), embedding_dimension,
                                      padding_idx=self.vocabulary["pad"])
        rnn_layer = torch.nn.LSTM if model_type == ModelType.LSTM else torch.nn.GRU
        self.rnn = rnn_layer(embedding_dimension, self.hidden_size, batch_first=True)
        self.relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=.3)
        self.linearA = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linearB = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size * 2, 1)
        # self.activation = torch.nn.Sigmoid()

    def forward(self, input_tensor, lengths):
        # embed on the device
        embedded = self.embedding(input_tensor)

        # # pack on the cpu, consider doing this so lstm ignores paddings
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)

        lstm_output, _ = self.rnn(packed)

        # # unpack on the cpu, consider doing this so lstm ignores paddings
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True,
                                                               padding_value=self.vocabulary["pad"])
        unpacked_final_state = unpacked[torch.arange(unpacked.size(0)), lengths - 1]

        linear_a_output = self.relu(self.linearA(unpacked_final_state))
        linear_b_output = self.relu(self.linearB(unpacked_final_state))
        combined_output = torch.cat((linear_a_output, linear_b_output), dim=1)
        return self.linear(combined_output)


class ConvoludedLSTMModel(nn.Module):
    def __init__(self, vocabulary, embedding_dimension, device="cpu", model_type=ModelType.LSTM, embedding=None):
        super(ConvoludedLSTMModel, self).__init__()
        self.device = device
        self.vocabulary = vocabulary
        self.hidden_size = 128
        self.embedding = embedding or nn.Embedding(self.vocabulary.__len__(), embedding_dimension,
                                      padding_idx=self.vocabulary["pad"])
        rnn_layer = torch.nn.LSTM if model_type == ModelType.LSTM else torch.nn.GRU
        self.rnn = rnn_layer(embedding_dimension, self.hidden_size, batch_first=True)
        self.conv1d_1 = torch.nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=9)
        self.layer_normalization = torch.nn.LayerNorm(self.hidden_size)
        # self.linearA = torch.nn.Linear(self.hidden_size, self.hidden_size * 2, device=self.device)
        # self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(self.hidden_size*4, 1)
        # self.activation = torch.nn.Sigmoid()

    def forward(self, input_tensor, lengths):
        # embed on the device
        embedded = self.embedding(input_tensor)

        # # pack on the cpu, consider doing this so lstm ignores paddings
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)

        lstm_output, _ = self.rnn(packed)
        # print(f"lstm_output shape: {lstm_output.shape}")

        # # unpack on the cpu, consider doing this so lstm ignores paddings
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True,
                                                             padding_value=self.vocabulary["pad"])
        # print(f"unpacked shape: {unpacked.shape}")

        permuted = unpacked.permute(0, 2, 1)
        # print(f"permuted shape: {permuted.shape}")

        convolved = self.conv1d(permuted)
        # print(f"convolved shape: {convolved.shape}")

        pooled_output = torch.nn.functional.adaptive_max_pool1d(convolved, 1).squeeze(dim=2)
        # print(f"pooled shape: {pooled_output}")

        # The output of the convolution are very large numbers
        # Normalize the outputs of each feature across all sequences
        normalized = self.layer_normalization(pooled_output)

        unpacked_final_state = unpacked[torch.arange(unpacked.size(0)), lengths - 1]
        linear_output = self.relu(self.linearA(unpacked_final_state))
        # print(f"linear shape: {linear_output}")

        # linear_a_output = self.relu(self.linearA(unpacked_final_state))
        # linear_b_output = self.linearB(self.relu(unpacked_final_state))
        combined_output = torch.cat((unpacked_final_state, linear_output, normalized), dim=1)
        return self.linear(combined_output)
