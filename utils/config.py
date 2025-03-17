import torch
import os
from torchvision.utils import save_image

# Configuration parameters
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)
EPOCHS = 10
IMG_SIZE = (512, 512)
CHECKPOINT_GEN = './checkpoints/gen_checkpoint.pth'
CHECKPOINT_DISC = './checkpoints/disc_checkpoint.pth'



# Ensure checkpoint directories exist
os.makedirs(os.path.dirname(CHECKPOINT_GEN), exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_DISC), exist_ok=True)

# Function to save the model checkpoint
def save_checkpoint(model, optimizer, filename):
    print(f"Saving checkpoint to {filename}...")
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename)

# Function to load the model checkpoint
def load_checkpoint(filename, model, optimizer, learning_rate, device='cpu'):
    print(f"Loading checkpoint from {filename}...")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Optionally adjust the learning rate after loading
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


# Function to save some example outputs from the generator
def save_some_examples(generator, data_loader, epoch, folder="examples"):
    print(f"Saving example images at epoch {epoch}...")
    generator.eval()
    os.makedirs(folder, exist_ok=True)
    # Select device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get a batch from the loader and pass it through the generator
    for idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        
        fake_images = generator(x)

        # Save the input (BW), target (colored), and fake images
        save_image(x, f"{folder}/bw_image_{epoch}_{idx}.png", normalize=True)  # Black & White input
        save_image(y, f"{folder}/target_image_{epoch}_{idx}.png", normalize=True)  # Target colored image
        save_image(fake_images, f"{folder}/fake_image_{epoch}_{idx}.png", normalize=True)  # Generated image
        
        # Save only a few examples (change `break` to save more)
        if idx == 20:
            break

    generator.train()  # Set the model back to training mode 