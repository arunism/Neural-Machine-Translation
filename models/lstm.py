import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseEncoder, BaseDecoder, BaseModel

class LstmEncoder(BaseEncoder):
    def __init__(self, config, input_size) -> None:
        super(LstmEncoder, self).__init__(config, input_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.layers_count, dropout=self.config.DROPOUT)
    
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        print(embedding)
        print(len(embedding))
        print(len(embedding[0][0]))
        output, (hidden, cell) = self.lstm(embedding)
        return hidden, cell


class LstmDecoder(BaseDecoder):
    def __init__(self, config, input_size, output_size) -> None:
        super(LstmDecoder, self).__init__(config, input_size, output_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.layers_count, dropout=self.config.DROPOUT)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, hidden, cell):
        # x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x)).view(1, 1, -1)
        output, (hidden, cell) = self.lstm(embedding, hidden, cell)
        prediction = self.fc(output)
        prediction = prediction.squeeze(0)
        return prediction, hidden, cell


class LstmModel(BaseModel):
    def __init__(self, config, input_size, output_size) -> None:
        super(LstmModel, self).__init__(config, input_size, output_size)
        self.encoder = LstmEncoder(self.config, self.input_size)
        self.decoder = LstmDecoder(self.config, self.input_size, self.output_size)
        self.encoder_optimizer = self.get_optimizer(self.encoder_optimizer_name, self.encoder)
        self.decoder_optimizer = self.get_optimizer(self.decoder_optimizer_name, self.decoder)

    def forward(self, src_tensor, target_tensor, tf=0.5):
        # self.encoder_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        target_len = target_tensor.shape[0]
        encoder_hidden, encoder_cell = self.encoder(src_tensor)
        outputs = torch.zeros(target_len, self.batch_size, self.output_size).to(self.device)
        x = target_tensor[0]
        for i in range(target_len):
            output, hidden, cell = self.decoder(x, encoder_hidden, encoder_cell)
            outputs[i] = output[0, 0]
        print(outputs)