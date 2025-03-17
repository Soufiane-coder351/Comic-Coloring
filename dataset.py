from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision
import pandas as pd

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class DatasetLoader(Dataset):
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
  

##############################################
# Test the DatasetLoader class               # 
##############################################


def test_dataset():
    # Define image size and dataset path
    img_size = (512, 512)  # Adjust based on your model's input size
    img_dir = "./Dataset"  # Root directory of dataset

    # Load dataset
    dataset = DatasetLoader(img_dir, img_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get one batch
    bw_img, color_img = next(iter(dataloader))

    # Convert tensors to numpy for visualization
    bw_img = bw_img.squeeze(0).permute(1, 2, 0).numpy()  # Remove batch dim, convert to HxWxC
    color_img = color_img.squeeze(0).permute(1, 2, 0).numpy()

    # Rescale from [-1, 1] to [0, 1] for display
    bw_img = (bw_img + 1) / 2
    color_img = (color_img + 1) / 2

    # Display images
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(bw_img.squeeze(), cmap="gray")
    ax[0].set_title("Black & White Image")
    ax[0].axis("off")

    ax[1].imshow(color_img)
    ax[1].set_title("Color Image")
    ax[1].axis("off")

    plt.show()


if __name__ == "__main__":
  # Run the test
  test_dataset()
