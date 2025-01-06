import pandas as pd
import torch
from dataloader import TrainDataset

# Load the dataset CSV file
dataFrame = pd.read_csv('./Dataset.csv')

# Assume you have your TrainDataset class already defined
dataset = TrainDataset('./Dataset', (64, 64))

# Iterate over each entry in the dataset DataFrame
for index in range(len(dataFrame)):
    # Retrieve grayscale and color images
    image_bw, image_color = dataset[index]
    
    # Print the shapes of the images
    print(f"Sample index {index}:")
    print("  Grayscale image shape:", image_bw.shape[0])  # Should be (1, 512, 512)
    print("  Color image shape:", image_color.shape[0])    # Should be (3, 512, 512)
    print()
    if image_bw.shape[0] != 1 or image_color.shape[0] != 3:
        print("Error: Unexpected image shape.")
        print("Error found at index ", index)
        print(image_bw.shape[0])
        print(image_color.shape[0])
        break