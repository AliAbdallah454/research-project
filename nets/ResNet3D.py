import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torchinfo
#from ResNet3D_moduls import generate_model


img_size = 256
batch_size = 16
epochs = 30
max_seq_length = 10
num_features = 512


class ResNet3D(nn.Module):
    def __init__(self, n_classes=4, num_frames=10, shape=1024, freeze_layers=10):
        super(ResNet3D, self).__init__()
        
        self.feature_extractor = models.video.r3d_18(pretrained=True)

        freeze_counter = 0
        for name, layer in self.feature_extractor.named_modules():
            if isinstance(layer, nn.Conv3d):
                freeze_counter += 1 
                if freeze_counter < 17:  
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.num_frames = num_frames
        self.shape = shape
        self.n_classes = n_classes
        self.feature_extractor.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_classes * num_frames)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, frames):
        #print(f'frames:{frames.shape}')
        frames = frames.permute(0, 2, 1, 3, 4)
        output = self.feature_extractor(frames)
        #output = self.feature_extractor.fc(frames)
        #print(f'output:{output.shape}')
        #output = output.view(-1, self.n_classes, self.num_frames)
        output = output.view(output.shape[0], self.num_frames, -1)
        #print(f'output after:{output.shape}')
        output1 = self.sigmoid(output)
        # output1 = output1.permute(0,2,1)
        return output1

    
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet3d = ResNet3D().to(device)
    #vid_input = torch.zeros(4,3,10,256,256).to(device)
    vid_input = torch.zeros(16,10,3,256,256).to(device)
    #output2, output1 = resnet3d(vid_input)
    output = resnet3d(vid_input)

    # print(f'seq_out_shape: {output2.shape}')
    # print(f'seq_out_shape: {output2.view(-1,4).shape}')
    # print("Feature Extractor Summary:")
    #print(summary(resnet3d, (3, 10, 256, 256), device='cuda'))
    # print(summary(resnet3d, (10,3, 256, 256), device='cuda'))
    print(output.shape)
    # print(resnet3d.modules)
    # for name, layer in resnet3d.named_modules():
    #     if isinstance(layer, nn.Conv3d):
    #         print(name, layer)

    #torchinfo.summary(sequence_model, (1, 10, 1024), device="cuda")