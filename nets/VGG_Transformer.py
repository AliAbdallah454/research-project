import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models
import torchinfo

img_size = 256
batch_size = 16
epochs = 30
max_seq_length = 10
num_features = 1024


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True)

        for param in vgg.features[:-10].parameters():
            param.requires_grad = False

        self.features = vgg.features
        self.Channel_reduce_conv = nn.Conv2d(512, 128, 1)
        self.pooling = nn.AdaptiveAvgPool2d((8, 8))
        self.dropout = nn.Dropout(0.2)
        self.rel1 = nn.ReLU()
        self.dense1 = nn.Linear(8 * 8 * 128, 1024)
        self.norm1 = nn.LayerNorm(1024)
        self.rel2 = nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = self.Channel_reduce_conv(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.rel1(self.dropout(x))
        x = self.dense1(x)
        x = self.rel2(self.norm1(x))
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, output_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(sequence_length, output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def forward(self, inputs):
        batch_size, length, num_features = inputs.size()
        positions = torch.arange(0, length).unsqueeze(0).repeat(batch_size, 1).cuda()
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dense_dim, num_classes):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.GELU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.dense_proj2 = nn.Linear(embed_dim, 64)
        self.dropout = nn.Dropout(0.2)
        self.dense_proj3 = nn.Linear(64, num_classes)

    def forward(self, inputs, mask=None):
        attention_output, _ = self.attention(inputs, inputs, inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        proj_output2 = self.layernorm_2(proj_output)
        proj_output2 = self.dense_proj2(proj_output2)
        proj_output2 = self.dropout(proj_output2)
        proj_output2 = self.dense_proj3(proj_output2)
    
        return proj_output2


class CNN_Transformer(nn.Module):
    def __init__(self, sequence_length=max_seq_length, embed_dim=num_features, num_heads=8, dense_dim=16, n_classes=3):
        super(CNN_Transformer, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.position_embedding = PositionalEmbedding(sequence_length, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, dense_dim, n_classes)
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim 
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.n_classes = n_classes
        self.sigmoid = nn.Sigmoid()

    def forward(self, frames):
        output = torch.zeros(frames.shape[0], self.sequence_length, self.embed_dim).cuda()

        for j in range(frames.shape[0]):
            for i in range (self.sequence_length):
                frames1 = frames[j,i,:,:,:].unsqueeze(0)
                output [j,i,:] = self.feature_extractor(frames1)      
        output1 = self.position_embedding(output)
        output1 = self.transformer_encoder(output1)
        output1 = self.sigmoid(output1)

        return output1
            

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor().to(device)
    img_input = torch.zeros(1,3,512,512).to(device)
    output = feature_extractor(img_input)
    sequence_model = TransformerEncoder(embed_dim=num_features, num_heads=8, dense_dim=16, num_classes=4).to(device)
    seq_input = torch.zeros(10, 1024).to(device)
    sequence_output = sequence_model(seq_input)
    # print(f'seq_out_shape: {sequence_output.shape}')
    # print("Feature Extractor Summary:")
    # print(summary(feature_extractor, (3, img_size, img_size), device=device))
    # torchinfo.summary(sequence_model, (1, 10, 1024), device=device)
    cnn_lstm = CNN_Transformer(10,1024).to(device)
    video_input = torch.zeros(16, 10, 3, 256, 256).to(device)
    video_output = cnn_lstm(video_input)
    print(video_output.shape)
