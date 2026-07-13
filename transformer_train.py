"""
transformer_train.py — 

"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformer_dataset import create_dataloaders, idx_to_square
from transformer_model import (build_transformer, causal_mask,
                                count_parameters, PAD_IDX, SOS_IDX, N_SQUARES)


def topk_accuracy(logits, targets, k):
    _, topk = torch.topk(logits, k, dim=-1)
    correct = topk.eq(targets.unsqueeze(1).expand_as(topk))
    return correct.any(dim=1).float().mean().item()


def chessboard_distance(idx_pred, idx_true):
    """Distance between two squares."""
    fp, rp = idx_pred % 8, idx_pred // 8
    ft, rt = idx_true % 8, idx_true // 8
    return float(max(abs(fp - ft), abs(rp - rt)))


def displacement_error(logits, targets):
    """
    ADE — Average Displacement Error.
    0 = perfect, 7 = max wrong.
    """
    preds = logits.argmax(dim=-1)
    total = sum(chessboard_distance(p.item(), t.item())
                for p, t in zip(preds, targets))
    return total / len(targets)


def final_displacement_error(logits, targets):
    """
    FDE — Final Displacement Error.
    For single-step prediction FDE = ADE.
    """
    return displacement_error(logits, targets)


def make_masks(src, device):
    """
     source mask.
    src: (batch, seq_len) — PAD token = PAD_IDX (64)
    """
    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(1).int().to(device)
    return src_mask


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = top1 = top3 = top5 = ade = 0.0
    n = len(loader)

    for src, tgt_label in loader:
        src       = src.to(device)
        tgt_label = tgt_label.to(device)   # (batch,) — indexes

        # decored input
        tgt_in = torch.full((src.size(0), 1), SOS_IDX,
                            dtype=torch.long, device=device)

        src_mask = make_masks(src, device)
        tgt_mask = causal_mask(tgt_in.size(1)).to(device)

        # Forward pass
        enc_out = model.encode(src, src_mask)
        dec_out = model.decode(enc_out, src_mask, tgt_in, tgt_mask)

        # last token 
        logits = model.project(dec_out[:, -1])  # (batch, N_SQUARES)

        # NLLLoss 
        loss = criterion(logits, tgt_label)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        top1 += topk_accuracy(logits.detach(), tgt_label, 1)
        top3 += topk_accuracy(logits.detach(), tgt_label, 3)
        top5 += topk_accuracy(logits.detach(), tgt_label, 5)
        ade  += displacement_error(logits.detach(), tgt_label)

    return total_loss/n, top1/n, top3/n, top5/n, ade/n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = top1 = top3 = top5 = ade = 0.0
    n = len(loader)

    for src, tgt_label in loader:
        src       = src.to(device)
        tgt_label = tgt_label.to(device)

        tgt_in = torch.full((src.size(0), 1), SOS_IDX,
                            dtype=torch.long, device=device)

        src_mask = make_masks(src, device)
        tgt_mask = causal_mask(tgt_in.size(1)).to(device)

        enc_out = model.encode(src, src_mask)
        dec_out = model.decode(enc_out, src_mask, tgt_in, tgt_mask)
        logits  = model.project(dec_out[:, -1])

        total_loss += criterion(logits, tgt_label).item()
        top1 += topk_accuracy(logits, tgt_label, 1)
        top3 += topk_accuracy(logits, tgt_label, 3)
        top5 += topk_accuracy(logits, tgt_label, 5)
        ade  += displacement_error(logits, tgt_label)

    return total_loss/n, top1/n, top3/n, top5/n, ade/n


def train(config):
    csv_path   = config['csv_path']
    seq_len    = config.get('seq_len',         20)
    batch_size = config.get('batch_size',      32)
    epochs     = config.get('num_epochs',     100)
    lr         = config.get('learning_rate', 3e-4)
    save_dir   = config.get('save_dir', 'checkpoints')
    patience   = config.get('patience',        20)

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    print("\nLoading data...")
    train_l, val_l, test_l = create_dataloaders(
        csv_path, seq_len=seq_len,
        batch_size=batch_size, num_workers=0,
        test_games=config.get('test_games', 5))

    # Model
    print("\nModel (Encoder-Decoder Transformer):")
    model = build_transformer(
        source_context_size = seq_len,
        target_context_size = 2,
        model_dimension     = config.get('model_dimension', 128),
        number_of_blocks    = config.get('number_of_blocks', 3),
        heads               = config.get('heads', 4),
        dropout             = config.get('dropout', 0.1),
        feed_forward_dimension = config.get('feed_forward_dimension', 256)
    ).to(device)
    count_parameters(model)

    # CrossEntropyLoss with label_smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-9)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    # Train
    print(f"\nTraining: {epochs} epochs | batch={batch_size} | lr={lr}")
    print("=" * 80)
    print(f"{'Ep':>4}  {'TrLoss':>7}  {'Tr@1':>5}  {'Tr@3':>5}  {'ADE':>5}  "
          f"{'VlLoss':>7}  {'Vl@1':>5}  {'Vl@3':>5}  {'ADE':>5}  {'s':>5}")
    print("-" * 80)

    best_val_top1    = 0.0
    patience_counter = 0
    history = {k: [] for k in ['train_loss','val_loss',
                                 'train_top1','val_top1',
                                 'train_top3','val_top3',
                                 'train_ade','val_ade']}

    for ep in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr1, tr3, tr5, tr_ade = train_epoch(
            model, train_l, optimizer, criterion, device)
        vl_loss, vl1, vl3, vl5, vl_ade = eval_epoch(
            model, val_l, criterion, device)

        scheduler.step()

        for k, v in zip(['train_loss','val_loss','train_top1',
                          'val_top1','train_top3','val_top3',
                          'train_ade','val_ade'],
                         [tr_loss, vl_loss, tr1, vl1, tr3, vl3,
                          tr_ade, vl_ade]):
            history[k].append(v)

        elapsed = time.time() - t0
        marker  = ''

        if vl1 > best_val_top1:
            best_val_top1    = vl1
            patience_counter = 0
            torch.save({
                'epoch':            ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_top1':         vl1,
                'val_top3':         vl3,
                'config':           config,
            }, os.path.join(save_dir, 'best_transformer.pth'))
            marker = ' ✓'
        else:
            patience_counter += 1

        if ep % 5 == 0 or marker or ep == 1:
            print(f"{ep:4d}  {tr_loss:7.4f}  {tr1:5.3f}  {tr3:5.3f}  {tr_ade:5.2f}  "
                  f"{vl_loss:7.4f}  {vl1:5.3f}  {vl3:5.3f}  {vl_ade:5.2f}  "
                  f"{elapsed:5.1f}{marker}")

        if patience_counter >= patience:
            print(f"\nEarly stopping — epoch {ep}.")
            break

    # Test
    print("\n" + "=" * 80)
    ckpt = torch.load(os.path.join(save_dir, 'best_transformer.pth'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    te_loss, te1, te3, te5, te_ade = eval_epoch(model, test_l, criterion, device)
    print(f"Test Loss:      {te_loss:.4f}")
    print(f"Test Top-1 Acc: {te1*100:.1f}%   (random baseline = 1.6%)")
    print(f"Test Top-3 Acc: {te3*100:.1f}%")
    print(f"Test Top-5 Acc: {te5*100:.1f}%")
    print(f"Test ADE:       {te_ade:.2f} squares  (0=perfect, 7=max mistake)")
    te_fde = te_ade  # single-step: FDE = ADE
    print(f"Test FDE:       {te_fde:.2f} squares  (Final Displacement Error)")

    # Examples
    print("\nPrediction Examples(Top-3):")
    model.eval()
    x_sample, y_sample = next(iter(test_l))
    x_sample = x_sample.to(device)
    tgt_in   = torch.full((x_sample.size(0), 1), SOS_IDX,
                          dtype=torch.long, device=device)
    src_mask = make_masks(x_sample, device)
    tgt_mask = causal_mask(1).to(device)

    with torch.no_grad():
        enc_out = model.encode(x_sample, src_mask)
        dec_out = model.decode(enc_out, src_mask, tgt_in, tgt_mask)
        logits  = model.project(dec_out[:, -1])

    probs = torch.softmax(logits, dim=-1)

    for i in range(min(5, len(y_sample))):
        true_sq  = idx_to_square(y_sample[i].item())
        top3_idx = torch.topk(probs[i], 3).indices.tolist()
        top3_sq  = [idx_to_square(j) for j in top3_idx]
        hit      = '✓' if true_sq in top3_sq else '✗'
        # Sequence 
        seq = [idx_to_square(t.item()) for t in x_sample[i]
               if t.item() != 64][-5:]
        print(f"  {hit} Gaze: ...{seq}  →  True: {true_sq}  "
              f"Pred: {top3_sq}")

    torch.save(history, os.path.join(save_dir, 'transformer_history.pth'))
    print(f"\nModel saved: {save_dir}/best_transformer.pth")
    return model, history


if __name__ == '__main__':
    config = {
        'csv_path':               'data/gaze_dataset.csv',
        'seq_len':                20,
        'batch_size':             16,
        'num_epochs':            300,
        'learning_rate':         3e-4,
        'save_dir':              'checkpoints',
        'patience':               40,
        # 2024 moves, 44 games
        'model_dimension':        64,
        'number_of_blocks':        2,
        'heads':                   4,
        'dropout':               0.2,
        'feed_forward_dimension': 256,
        'test_games':             12,    # ~260 test moves
    }

    model, history = train(config)