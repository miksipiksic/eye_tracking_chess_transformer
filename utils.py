"""
utils.py 

  1. visualize_predictions  — predicted pupil vs exact pupil
  2. plot_history           — training curves
  3. screen_to_square       — screen coords -> board square
  4. gaze_to_fixations      — filter to fixation
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ORIG_SIZE = 416.0


@torch.no_grad()
def visualize_predictions(model, loader, device,
                           n=8, save_path=None):

    model.eval()

    imgs, labels = next(iter(loader))
    imgs   = imgs.to(device)
    preds  = model(imgs).cpu()
    imgs   = imgs.cpu()

    img_size = imgs.shape[-1]
    scale = img_size   

    n   = min(n, len(imgs))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]

        # De-normalization
        img = imgs[i, 0].numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)

        # Coords based on img_size
        def coords(t):
            return t[0].item() * scale, t[1].item() * scale, \
                   t[2].item() * scale

        tx, ty, tr = coords(labels[i])
        px, py, pr = coords(preds[i])

        # Exact pupil - green dot
        ax.add_patch(patches.Circle((tx, ty), tr,
                     color='lime', fill=False, lw=1.8, label='Tačno'))
        # Predicted pupil - red circle
        ax.add_patch(patches.Circle((px, py), pr,
                     color='red', fill=False, lw=1.8,
                     linestyle='--', label='Pred.'))

        err = np.sqrt((px - tx)**2 + (py - ty)**2)
        ax.set_title(f'Δ={err:.1f}px', fontsize=9)
        ax.axis('off')

    axes[0].legend(loc='upper right', fontsize=7, framealpha=0.7)

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle(
        'Pupil — green: correct  |  red: prediction\n'
        'Δ = error in pixels',
        fontsize=10
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved: {save_path}")
    else:
        plt.show()
    plt.close()



def plot_history(history, save_path=None):

    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], 'b-o', ms=3, label='Train')
    ax1.plot(epochs, history['val_loss'],   'r-o', ms=3, label='Val')
    ax1.set_title('MSE Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_px'], 'b-o', ms=3, label='Train')
    ax2.plot(epochs, history['val_px'],   'r-o', ms=3, label='Val')
    ax2.set_title('Pupil error (pixels)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pixel error')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle('Training flow', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graph saved: {save_path}")
    else:
        plt.show()
    plt.close()



def screen_to_square(gaze_x, gaze_y,
                     board_x, board_y, board_w, board_h,
                     flipped=False):
 
    rel_x = gaze_x - board_x
    rel_y = gaze_y - board_y

    if not (0 <= rel_x < board_w and 0 <= rel_y < board_h):
        return {'square': None, 'col': None, 'row': None, 'valid': False}

    col = min(int(rel_x / (board_w / 8)), 7)
    row = min(int(rel_y / (board_h / 8)), 7)

    # Flipped: 
    if flipped:
        col = 7 - col
        row = 7 - row

    square = f"{chr(ord('a') + col)}{8 - row}"
    return {'square': square, 'col': col, 'row': row, 'valid': True}



def gaze_to_fixations(gaze_sequence, board_params, min_dwell_ms=80):
   
    fixations    = []
    cur_sq       = None
    sq_start     = None
    visit_count  = 0

    for pt in gaze_sequence:
        sq = screen_to_square(pt['x'], pt['y'], **board_params)

        if not sq['valid']:
            # looking outside of the board
            if cur_sq is not None:
                dwell = pt['t'] - sq_start
                if dwell >= min_dwell_ms:
                    fixations.append({
                        'square':   cur_sq,
                        'dwell_ms': dwell,
                        'visits':   visit_count
                    })
            cur_sq = None
            continue

        if sq['square'] != cur_sq:
            if cur_sq is not None:
                dwell = pt['t'] - sq_start
                if dwell >= min_dwell_ms:
                    fixations.append({
                        'square':   cur_sq,
                        'dwell_ms': dwell,
                        'visits':   visit_count
                    })
            cur_sq      = sq['square']
            sq_start    = pt['t']
            visit_count = 1
        else:
            visit_count += 1

    return fixations

# Test

if __name__ == '__main__':
    print("Test screen_to_square")
    print("=" * 46)

    bp = {'board_x': 100, 'board_y': 50,
          'board_w': 800, 'board_h': 800}

    cases = [
        (100,  50,  'a8'),   
        (899,  50,  'h8'),   
        (100,  849, 'a1'),   
        (899,  849, 'h1'),   
        (500,  450, 'e4'),    
        (50,   50,  None),   
    ]

    ok = True
    for x, y, exp in cases:
        res = screen_to_square(x, y, **bp)
        got = res['square']
        sym = '✓' if got == exp else '✗'
        if got != exp:
            ok = False
        print(f"  {sym}  ({x:4d},{y:4d}) → {str(got):4}  "
              f"(expected: {str(exp):4})")

    print(f"\n{'Tests passed!' if ok else 'MISTAKES AVAILABLE!'}")