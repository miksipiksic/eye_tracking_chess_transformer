"""
train.py — train cnn

  - GlobalAvgPool umesto Flatten (manji model)
  - Dropout (0.5)
  - Epoch (100)
  - LR warmup first 5, than ReduceLROnPlateau

Goal: Val px < 8 on image 256×256.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import create_dataloaders
from model import GazeCNN, count_parameters

ORIG_SIZE = 256.0


def pixel_error(pred, target):
    """Pupil center Euclid distance in pixels."""
    px = pred[:, :2]   * ORIG_SIZE
    tx = target[:, :2] * ORIG_SIZE
    return torch.sqrt(((px - tx) ** 2).sum(dim=1)).mean().item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_px, n = 0.0, 0.0, len(loader)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_px   += pixel_error(preds.detach(), labels.detach())
    return total_loss / n, total_px / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_px, n = 0.0, 0.0, len(loader)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        total_loss += criterion(preds, labels).item()
        total_px   += pixel_error(preds, labels)
    return total_loss / n, total_px / n


def train(config):
    npz_path   = config['npz_path']
    csv_path   = config['csv_path']
    img_size   = config.get('img_size',       128)
    batch_size = config.get('batch_size',      16)
    epochs     = config.get('num_epochs',     100)
    lr         = config.get('learning_rate', 5e-4)
    save_dir   = config.get('save_dir', 'checkpoints')
    patience   = config.get('patience',        20)

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Podaci ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        npz_path=npz_path, csv_path=csv_path,
        img_size=img_size, batch_size=batch_size,
        num_workers=0
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print("\nModel:")
    model = GazeCNN(img_size=img_size, dropout=0.5).to(device)
    count_parameters(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=10)

    # ── Trening ─────────────────────────────────────────────────────────────
    print(f"\nTraining: {epochs} epoch | batch={batch_size} | lr={lr}")
    print("=" * 72)
    print(f"{'Ep':>4}  {'Tr loss':>8}  {'Tr px':>7}  "
          f"{'Vl loss':>8}  {'Vl px':>7}  {'LR':>8}  {'s':>5}")
    print("-" * 72)

    best_val_loss    = float('inf')
    patience_counter = 0
    history          = {'train_loss': [], 'val_loss': [],
                        'train_px':   [], 'val_px':   []}

    for ep in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_px = train_epoch(model, train_loader,
                                     optimizer, criterion, device)
        vl_loss, vl_px = eval_epoch(model, val_loader,
                                    criterion, device)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_px'].append(tr_px)
        history['val_px'].append(vl_px)

        lr_now  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        marker  = ''

        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            patience_counter = 0
            torch.save({
                'epoch':            ep,
                'model_state_dict': model.state_dict(),
                'val_loss':         vl_loss,
                'val_px':           vl_px,
                'config':           config,
            }, os.path.join(save_dir, 'best_model.pth'))
            marker = ' ✓'
        else:
            patience_counter += 1

        # print epoch if the results are bettwr (interval = 5)
        if ep % 5 == 0 or marker or ep == 1:
            print(f"{ep:4d}  {tr_loss:8.5f}  {tr_px:7.2f}  "
                  f"{vl_loss:8.5f}  {vl_px:7.2f}  "
                  f"{lr_now:8.6f}  {elapsed:5.1f}{marker}")

        if patience_counter >= patience:
            print(f"\nEarly stopping — epoha {ep}.")
            break

    #Test
    print("\n" + "=" * 72)
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    te_loss, te_px = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss:        {te_loss:.5f}")
    print(f"Test Pixel Error: {te_px:.2f} px  (om image 256×256)")

    torch.save(history, os.path.join(save_dir, 'history.pth'))
    print(f"\nModel saved {save_dir}/best_model.pth")
    return model, history


if __name__ == '__main__':
    config = {
        'npz_path':      'data/eye_dataset.npz',
        'csv_path':      'data/annotation.csv',
        'img_size':      128,
        'batch_size':    16,    # small dataset
        'num_epochs':    100,   
        'learning_rate': 5e-4,  # stability
        'save_dir':      'checkpoints',
        'patience':      20,    #  early stopping
    }
    model, history = train(config)