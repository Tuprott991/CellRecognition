import torch
import torch.nn as nn
from dataset import LIVECellDataset
from model_arch import DeepLabV3Plus
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        preds = preds.squeeze(1)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).squeeze(1)
        loss = criterion(preds, masks)
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# --- Main ---
def main():
    # Paths
    train_img_dir = '/path/to/livecell/train/images'
    train_mask_dir = '/path/to/livecell/train/masks'
    val_img_dir = '/path/to/livecell/val/images'
    val_mask_dir = '/path/to/livecell/val/masks'

    # Hyperparameters
    batch_size = 8
    lr = 1e-4
    epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])

    # Datasets & Loaders
    train_ds = LIVECellDataset(train_img_dir, train_mask_dir, transform)
    val_ds = LIVECellDataset(val_img_dir, val_mask_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    model = DeepLabV3Plus('convnextv2_small', num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_deeplabv3plus_livecell.pth')

if __name__ == '__main__':
    main()
