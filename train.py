import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DatasetLoader
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import save_checkpoint, load_checkpoint, save_some_examples, CHECKPOINT_GEN, CHECKPOINT_DISC, LEARNING_RATE, EPOCHS

# Select device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fn(disc, gen, loader, opt_disc, opt_gen, BCE, L1_Loss):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop): # x is BW and y is colored
        x, y = x.to(device), y.to(device)
        # Train Discriminator
        y_fake = gen(x)
        D_real = disc(x, y)
        D_fake = disc(x, y_fake.detach())
        D_real_loss = BCE(D_real, torch.ones_like(D_real))
        D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train Generator
        D_fake = disc(x, y_fake)
        G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
        L1 = L1_Loss(y_fake, y) * 100
        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

def main():
    disc = Discriminator(in_channel=3).to(device)
    gen = Generator(in_channels=1).to(device)

    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    # Load checkpoints if needed
    # if config.LOAD_MODEL:
    #     load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE, device=device)
    #     load_checkpoint(CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE, device=device)

    train_dataset = DatasetLoader(img_dir="./Dataset", img_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

    for epoch in range(EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, BCE, L1_Loss)

        # Save the model and examples periodically
        if epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, CHECKPOINT_DISC)

        save_some_examples(gen, train_loader, epoch, folder="examples")

if __name__ == "__main__":
    main()
