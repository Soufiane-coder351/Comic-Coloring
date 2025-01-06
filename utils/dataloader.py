from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision
import pandas as pd
from torch import nn
import torch

class TrainDataset(Dataset):
    def __init__(self,img_dir,img_size):
        self.img_dir = img_dir
        self.img_size = img_size
        self.dataFrame = pd.read_csv('./Dataset.csv')

    def __len__(self):
        return len(self.dataFrame)
    
    def __getitem__(self, index):

        path_image_color = self.dataFrame.iloc[index,0]
        path_image_bw = self.dataFrame.iloc[index,1]
        
        image_color = read_image(path_image_color)
        image_bw = read_image(path_image_bw)

        image_color = torchvision.transforms.Resize(self.img_size)(image_color)
        image_bw = torchvision.transforms.Resize(self.img_size)(image_bw)

        # Ensure image_color has 3 channels
        if image_color.shape[0] != 3:
            image_color = image_color.repeat(3, 1, 1)  # Convert to RGB

        # Ensure image_bw has 1 channel
        if image_bw.shape[0] != 1:  
            image_bw = image_bw.mean(dim=0, keepdim=True)  # Convert to grayscale 

        # Normalize the images
        image_color = (image_color-127.5)/127.5
        image_bw = (image_bw-127.5)/127.5

        return image_bw,image_color


class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.couche1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1)
        self.couche2 = nn.BatchNorm2d(out_channels)
        self.couche3 = nn.LeakyReLU(0.2)

    def forward(self,x):
        x = self.couche1(x)
        x = self.couche2(x)
        x = self.couche3(x)
        return x


class Up(nn.Module):
    def __init__(self,in_channels,out_channels, activation=False):
        super().__init__()
        self.couche1 = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1, output_padding=1)
        self.couche2 = nn.BatchNorm2d(out_channels)
        self.couche3 = nn.LeakyReLU(0.2)
        if activation: 
            self.couche3 = nn.Tanh()

    def forward(self, x):
        x = self.couche1(x)
        x = self.couche2(x)
        x = self.couche3(x)
        return x
    


class AutoEncoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features 
        self.down1 = Down(1                  , self.num_features  ) 
        self.down2 = Down(self.num_features  , self.num_features*2)
        self.down3 = Down(self.num_features*2, self.num_features*4)
        self.down4 = Down(self.num_features*4, self.num_features*8)

        self.up1 = Up(self.num_features*8, self.num_features*4)
        self.up2 = Up(self.num_features*4, self.num_features*2)
        self.up3 = Up(self.num_features*2, self.num_features  )
        self.up4 = Up(self.num_features  ,                   3, True)

    def forward(self, x):
        x = self.down1(x)
#        print(x.size())
        x = self.down2(x)
#        print(x.size())
        x = self.down3(x)
#        print(x.size())
        x = self.down4(x)
#        print(x.size())

        x = self.up1(x)
#        print(x.size())
        x = self.up2(x)
#        print(x.size())
        x = self.up3(x)
#        print(x.size())
        x = self.up4(x)
#        print(x.size())
        return x

# test = TrainDataset('./Dataset', (64, 64))
# print(test.__getitem__(91)[0].shape)
# print(test.__getitem__(91)[1].shape)