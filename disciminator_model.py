import torch
import torch.nn as nn

class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=2):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2),
    )
    
  def forward(self, x):
    return self.conv(x)
    

# x,y <- concatenate these across the channels
class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        # initlal layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channel * 2, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        # CNN blocks
        self.conv2 = CNNBlock(64, 128, stride=2)
        self.conv3 = CNNBlock(128, 256, stride=2)
        self.conv4 = CNNBlock(256, 512, stride=1)

        # final layer (PatchGAN) -> 1 channel output
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.final(x)



def test():
  x = torch.randn((1, 3, 256, 256))
  y = torch.randn((1, 3, 256, 256))

  model = Discriminator()
  preds = model(x, y)
  print(preds.shape)


if __name__ == "__main__":
  test()