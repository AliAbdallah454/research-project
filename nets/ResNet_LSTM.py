import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torchinfo


img_size = 256
batch_size = 16
epochs = 30
max_seq_length = 10
num_features = 2048



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # for i in range(4, len(resnet.features)):
        #     for param in resnet.features[i].parameters():
        #         param.requires_grad = True

        # ct = 0
        # for child in resnet.children():
        #     ct += 1
        #     if ct < 4:
        #         for param in child.parameters():
        #             param.requires_grad = False
                
        self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        self.Channel_reduce_conv = nn.Conv2d(1024, 64, 1)
        self.pooling = nn.AdaptiveAvgPool2d((8, 8))
        self.dropout = nn.Dropout(0.4)
        self.dense1 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.resnet(x)
        x = self.Channel_reduce_conv(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.dense1(x)
        return x


# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         vgg = models.vgg16(pretrained=True)

#         for param in vgg.parameters():
#             param.requires_grad = False
#         # for i, param in enumerate(vgg.features.parameters()):
#         #     if i < 5:
#         #         param.requires_grad = False
#         #     else:
#         #         break
            
#         self.features = vgg.features
#         self.Channel_reduce_conv = nn.Conv2d(512, 128, 1)
#         self.pooling = nn.AdaptiveAvgPool2d((8, 8))
#         self.dropout = nn.Dropout(0.4)
#         self.dense1 = nn.Linear(8192, 1024)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.Channel_reduce_conv(x)
#         x = self.pooling(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.dropout(x)
#         x = self.dense1(x)
#         return x


# class FeatureExtractor(nn.Module):
#     def __init__(self, embed_dim=512, dropout=0.4):
#         super().__init__()
#         # 1) load pretrained ResNet50
#         resnet = models.resnet50(pretrained=True)
        
#         # 2) freeze everything except layer3 & layer4
#         for name, param in resnet.named_parameters():
#             param.requires_grad = False
#             if name.startswith("layer3") or name.startswith("layer4"):
#                 param.requires_grad = True

#         # 3) chop off the last two blocks (avgpool + fc)
#         #    so we get a [B, 2048, H, W] feature map
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
#         # 4) global‐avg‐pool → [B,2048]
#         self.gap = nn.AdaptiveAvgPool2d(1)
        
#         # 5) projection head: 2048 → embed_dim
#         self.fc      = nn.Linear(2048, embed_dim)

#         #self.bn      = nn.BatchNorm1d(embed_dim)
#         self.norm      = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: [B, 3, H, W]
#         x = self.backbone(x)          # → [B, 2048, h', w']
#         x = self.gap(x)               # → [B, 2048, 1, 1]
#         x = x.view(x.size(0), -1)     # → [B, 2048]
#         x = self.fc(x)                # → [B, embed_dim]
#         #x = self.bn(x)
#         x = self.norm(x) 
#         x = F.relu(x)
#         x = self.dropout(x)
#         return x 



class SequenceModel(nn.Module):
    def __init__(self, num_classes):
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.dense1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        # self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        # self.dropout2 = nn.Dropout(0.4)
        # self.dense2 = nn.Linear(256, 64)
        # self.dropout2 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(64, num_classes)


    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.relu(x)
        # x = self.dense2(x)
        # x = self.dropout2(x)
        x = self.dense2(x)

        return x

class CNN_LSTM(nn.Module):
    def __init__(self, n_classes, num_frames=10, shape=1024):
        super(CNN_LSTM, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.sequence_model = SequenceModel(n_classes)
        self.num_frames = num_frames
        self.shape = shape
        self.n_classes = n_classes
        self.sigmoid = nn.Sigmoid()


    def forward(self, frames):
        output = torch.zeros(frames.shape[0], self.num_frames, self.shape).cuda()

        for j in range(frames.shape[0]):
            for i in range (self.num_frames):
                output [j,i,:] = self.feature_extractor(frames[j,i,:,:].unsqueeze(0))
            
        output1 = self.sequence_model(output)
        output1 = self.sigmoid(output1)

        return output1
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor().to(device)
    img_input = torch.zeros(1,3,512,512).to(device)
    output = feature_extractor(img_input)
    sequence_model = SequenceModel(num_classes=2).to(device)
    seq_input = torch.zeros(10, 1024).to(device)
    sequence_output = sequence_model(seq_input)
    print(f'seq_out_shape: {sequence_output.shape}')

    print("Feature Extractor Summary:")
    print(summary(feature_extractor, (3, img_size, img_size), device='cuda'))

    torchinfo.summary(sequence_model, (1, 10, 1024), device="cuda")

    cnn_lstm = CNN_LSTM(2, 10,1024).to(device)
    video_input = torch.zeros(4, 10, 3, 256, 256).to(device)
    video_output = cnn_lstm(video_input)
    print(video_output.shape)

    