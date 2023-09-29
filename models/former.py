import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, output_size, seq_len):
        super(TransformerModel, self).__init__()

        # embed_dim = head_dim * num_heads?
        self.input_fc = nn.Linear(input_size, d_model)
        self.output_fc = nn.Linear(input_size, d_model)
        #         self.pos_emb = PositionalEncoding( d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
            device=device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dropout=0.1,
            dim_feedforward=4 * d_model,
            batch_first=True,
            device=device
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.fc = nn.Linear(output_size * d_model, output_size)
        self.fc1 = nn.Linear(seq_len * d_model, d_model)
        self.fc2 = nn.Linear(d_model, output_size)
        self.output_size = output_size

    def forward(self, x):
        # print(x.size())  # (256, 24, 7)
        y = x[:, - self.output_size:, :]
        # print(y.size())  # (256, 4, 7)
        x = self.input_fc(x)  # (256, 24, 128)
        #         x = self.pos_emb(x)   # (256, 24, 128)
        x = self.encoder(x)
        # 不经过解码器
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        out = self.fc2(x)
        # y = self.output_fc(y)   # (256, 4, 128)
        # out = self.decoder(y, x)  # (256, 4, 128)
        # out = out.flatten(start_dim=1)  # (256, 4 * 128)
        # out = self.fc(out)  # (256, 4)

        return out
