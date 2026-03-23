"""
transformer_dataset.py 
  CNN → (x,y) koordinate → TOKENIZATION → squares → Transformer → probabilites

Tokenization:

  64 squares (a1-h8) + 1 PAD token = 65 tokens
  a1=0, b1=1, ..., h8=63, PAD=64
Input: square index sequence  [28, 35, 36, 64, 64, ...]  (last 20)
IOutput: probability for each square
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

SEQ_LEN   = 20
PAD_IDX   = 64
N_SQUARES = 64


def square_to_idx(sq: str) -> int:
    """'a1' → 0, 'h8' → 63"""
    sq   = sq.strip()
    file = ord(sq[0]) - ord('a')
    rank = int(sq[1]) - 1
    return rank * 8 + file


def idx_to_square(idx: int) -> str:
    """0 → 'a1', 63 → 'h8'"""
    return f"{chr(ord('a') + idx % 8)}{idx // 8 + 1}"


def encode_sequence(seq_str: str, seq_len: int = SEQ_LEN) -> list:
    """
    String sequence -> token list (len = seq_len)
    Procces:
      1. Parsing: "e4,d5,e5" → ['e4','d5','e5']
      2. Sqaure to index: ['e4','d5','e5'] → [28, 35, 36]
      3. Take last seq_len tokena
      4. Fill with PAD

    Example
      "e4,d5,e5" → [64,64,...,28,35,36]  (17 PAD + 3 tokens)
    """
    fields = [f.strip() for f in seq_str.split(',') if f.strip()]

    # TOKENIZATION
    tokens = []
    for f in fields:
        try:
            tokens.append(square_to_idx(f))
        except Exception:
            continue

    # remove last 3 - mouse + played
    TRIM = 3
    if len(tokens) > TRIM:
        tokens = tokens[:-TRIM]
    tokens = tokens[-seq_len:] # last n

    # + pad
    return [PAD_IDX] * (seq_len - len(tokens)) + tokens


class GazeChessDataset(Dataset):
    """
    Dataset for Encoder-Decoder Transformer.

      x: tensor (seq_len,) — token indexes of coords
      y: int               — target square index (0–63)
    """

    def __init__(self, csv_path, seq_len=SEQ_LEN, augment=False):
        self.seq_len = seq_len
        self.augment = augment

        cols = ['gaze_sequence','move_uci','move_san','move_number',
                'color','fen','target_square','mode']
        df = pd.read_csv(csv_path, header=None, names=cols)
        df = df[df['gaze_sequence'].str.strip() != ''].copy()
        df = df[df['target_square'].str.strip() != ''].copy()

        valid = []
        for t in df['target_square']:
            try:
                square_to_idx(str(t).strip())
                valid.append(True)
            except Exception:
                valid.append(False)
        df = df[valid].reset_index(drop=True)

        self.sequences = df['gaze_sequence'].tolist()
        self.targets   = df['target_square'].str.strip().tolist()

        print(f"Dataset loaded: {len(self)} examples")
        print(f"  Seq len: {seq_len} tokens")
        print(f"  Classes:   {len(set(self.targets))}/64 squares")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_str = self.sequences[idx]
        target  = self.targets[idx]

        tokens = encode_sequence(seq_str, self.seq_len)

        # Augment, random seq shortening
        if self.augment:
            n_real = sum(1 for t in tokens if t != PAD_IDX)
            if n_real > 4:
                cut = np.random.randint(2, n_real + 1)
                tokens = [PAD_IDX] * (self.seq_len - cut) + \
                         [t for t in tokens if t != PAD_IDX][-cut:]

        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(square_to_idx(target), dtype=torch.long)
        return x, y


def create_dataloaders(csv_path, seq_len=SEQ_LEN, batch_size=32,
                       val_split=0.15, test_games=5,
                       num_workers=0, seed=42):
    """
    Split by games

    Game moves: last test_games games → test set
                before last test_games→ train/val (85/15)
    Puzzle: train/val (not in test)

    - model doesn't see test games before training
    """
    torch.manual_seed(seed)

    cols = ['gaze_sequence','move_uci','move_san','move_number',
            'color','fen','target_square','mode']
    df = pd.read_csv(csv_path, header=None, names=cols)
    df['move_number'] = pd.to_numeric(df['move_number'], errors='coerce')

    # games len (end)
    game_id = 0
    game_ids = [0]
    for i in range(1, len(df)):
        cur  = df['move_number'].iloc[i]
        prev = df['move_number'].iloc[i-1]
        mode = df['mode'].iloc[i]
        if mode == 'game' and cur <= 2 and prev > 5:
            game_id += 1
        game_ids.append(game_id)
    df['game_id'] = game_ids

    game_ids_unique = sorted(df[df['mode']=='game']['game_id'].unique())
    n_games = len(game_ids_unique)

    # last test_games games
    test_game_ids  = set(game_ids_unique[-test_games:])
    train_game_ids = set(game_ids_unique[:-test_games])

   
    test_idx  = df.index[
        (df['mode'] == 'game') & (df['game_id'].isin(test_game_ids))
    ].tolist()

    trainval_idx = df.index[
        (df['mode'] == 'game') & (df['game_id'].isin(train_game_ids))
        | (df['mode'] != 'game')  # puzzle uvek u train/val
    ].tolist()

    # Val = 15% trainval 
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(trainval_idx), generator=rng).tolist()
    n_val   = int(len(trainval_idx) * val_split)
    val_idx   = [trainval_idx[i] for i in perm[:n_val]]
    train_idx = [trainval_idx[i] for i in perm[n_val:]]

    print(f"\nGames split:")
    print(f"  Train: {len(train_idx):4d} moves (games 1-{n_games-test_games} + puzzle)")
    print(f"  Val:   {len(val_idx):4d} moves (15% train+puzzle)")
    print(f"  Test:  {len(test_idx):4d} moves  (games {n_games-test_games+1}-{n_games} — NEVER SEEN)")


    full_ds    = GazeChessDataset(csv_path, seq_len, augment=False)
    train_ds   = GazeChessDataset(csv_path, seq_len, augment=True)

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(Subset(full_ds, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(Subset(full_ds, test_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Test transformer_dataset.py")
    print("=" * 50)

    ds = GazeChessDataset('data/gaze_dataset.csv', seq_len=20)

    print("\nFirst 5 examples:")
    for i in range(min(5, len(ds))):
        x, y = ds[i]
        # decode back to sq (ignore pad)
        seq = [idx_to_square(t.item()) for t in x if t.item() != PAD_IDX]
        print(f"\nExample {i}:")
        print(f"  Input  (coords): {seq}")
        print(f"  Output (probabilities):         {idx_to_square(y.item())}")

    train_l, val_l, test_l = create_dataloaders(
        'data/gaze_dataset.csv', seq_len=20, batch_size=32)
    x_b, y_b = next(iter(train_l))
    print(f"\nBatch: x={x_b.shape}  y={y_b.shape}")
    print(f"x Range: [{x_b.min()}, {x_b.max()}]  (64=PAD)")
    print("\nOK!")