import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob, os, time
from dan_model import DAN

class ImageDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, '*.png')))
        self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.tf(img)

def train(epochs=30, batch_size=16, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    dataset = ImageDataset('dataset')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DAN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            recon, code = model(batch)
            loss = loss_fn(recon, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.5f}  elapsed={time.time()-start:.1f}s")

    torch.save(model.state_dict(), 'dan_model.pt')
    print(f"Saved model to dan_model.pt (training took {time.time()-start:.1f}s)")
    return model

if __name__ == '__main__':
    train()
