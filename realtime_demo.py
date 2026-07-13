"""
Real-time demo — eye-tracking/mouse mode + puzzle mode with prediction heatmaps.

Pipeline: webcam → pupil detection (MediaPipe / CNN / HoughCircles)
→ calibration → gaze position → square fixations → Transformer
→ probability heatmap of the next move's destination square.
"""

import pygame
import chess
import cv2
import numpy as np
import torch
import os, sys, time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import build_transformer, causal_mask, SOS_IDX, PAD_IDX
from transformer_dataset import encode_sequence, idx_to_square, SEQ_LEN
from utils import screen_to_square
from model import GazeCNN
import ui_theme as ui

# Layout
MARGIN      = 24
BOARD_X     = MARGIN
BOARD_Y     = ui.HEADER_H + 20
BOARD_SIZE  = 560
SQUARE_SIZE = BOARD_SIZE // 8

PANEL_X     = BOARD_X + BOARD_SIZE + 24
CAM_CARD_W  = 234
CAM_W       = 210            # displayed video width
CAM_VID_H   = 158            # displayed video height (4:3)
CAM_CARD_H  = 224
GAP         = 12
INFO_X      = PANEL_X + CAM_CARD_W + GAP
INFO_W      = 210
PANEL_W     = CAM_CARD_W + GAP + INFO_W

CHART_Y     = BOARD_Y + CAM_CARD_H + GAP
CHART_H     = BOARD_SIZE - CAM_CARD_H - GAP

WINDOW_W    = INFO_X + INFO_W + MARGIN
WINDOW_H    = BOARD_Y + BOARD_SIZE + 44

TRANSFORMER_PATH = 'checkpoints/best_transformer.pth'
CNN_PATH         = 'checkpoints/best_model.pth'
CALIB_FILE       = 'checkpoints/calibration.npy'

GAZE_WINDOW  = 20
TRIM         = 3
MIN_DWELL    = 8      # frames on the same square = fixation
UPDATE_EVERY = 10     # frames between predictions (faster update)
SMOOTH_N     = 25     # moving average pupil coords (more -> more stable)

PIECES = {
    chess.PAWN:   {chess.WHITE:'♙',chess.BLACK:'♟'},
    chess.KNIGHT: {chess.WHITE:'♘',chess.BLACK:'♞'},
    chess.BISHOP: {chess.WHITE:'♗',chess.BLACK:'♝'},
    chess.ROOK:   {chess.WHITE:'♖',chess.BLACK:'♜'},
    chess.QUEEN:  {chess.WHITE:'♕',chess.BLACK:'♛'},
    chess.KING:   {chess.WHITE:'♔',chess.BLACK:'♚'},
}

# Puzzles if there is no csv with lichess puzzles
FALLBACK_PUZZLES = [
    {'fen':'4k3/8/4K3/4R3/8/8/8/8 w - - 0 1','solution':'e5e8','desc':'Mate in 1 — rook to e8'},
    {'fen':'6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1','solution':'e1e8','desc':'Rook to the 8th rank'},
    {'fen':'r3k2r/ppp2ppp/2n5/3p4/3P4/2N5/PPP2PPP/R4RK1 w kq - 0 1','solution':'f1f8','desc':'Rook attack'},
    {'fen':'8/8/8/3k4/8/3K4/8/4R3 w - - 0 1','solution':'e1e5','desc':'Rook to the 5th rank'},
    {'fen':'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
     'solution':'f3g5','desc':'Attack on f7'},
]

# Moving average filter for coordinate stabilization

class GazeFilter:
    """
    Median + Moving Average filter
    Median is more resilient to outliers than moving average
    """
    def __init__(self, n=SMOOTH_N):
        self.xs = deque(maxlen=n)
        self.ys = deque(maxlen=n)

    def update(self, x, y):
        # remove extreme outliers
        if len(self.xs) > 5:
            mx = np.median(list(self.xs))
            my = np.median(list(self.ys))
            # 80 px threshold
            if abs(x - mx) > 80 or abs(y - my) > 80:
                return
        self.xs.append(x)
        self.ys.append(y)

    def get(self):
        if len(self.xs) < 3:
            return None, None
        # median
        return int(np.median(list(self.xs))), int(np.median(list(self.ys)))

    def reset(self):
        self.xs.clear(); self.ys.clear()

# pupil detection

_EYE_CASCADE = None
def get_cascade():
    global _EYE_CASCADE
    if _EYE_CASCADE is None:
        _EYE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
    return _EYE_CASCADE

from torchvision import transforms as T
_CNN_TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# MediaPipe Face Mesh for pupil detection
_mp_face_mesh = None
_mp_drawing   = None

def get_mediapipe():
    global _mp_face_mesh, _mp_drawing
    if _mp_face_mesh is None:
        try:
            import mediapipe as mp
            _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,   # iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Face Mesh loaded.")
        except ImportError:
            print("MediaPipe not installed — using HoughCircles.")
            _mp_face_mesh = False
    return _mp_face_mesh if _mp_face_mesh else None


def detect_pupil_tracking(frame_bgr):
    """
    Pupil detection for calibration and training

      1. MediaPipe Face Mesh — iris landmarks
      2. HoughCircles fallback

    Returns (cx, cy) coordinates in camera pixels

    """
    if frame_bgr is None:
        return None, None

    H, W = frame_bgr.shape[:2]

    # MediaPipe
    mp_mesh = get_mediapipe()
    if mp_mesh is not None:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = mp_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                r_iris = lm[468]   # right eye iris center
                l_iris = lm[473]   # left eye iris center
                # choosing the one that's more centered (horizontaly)
                r_x = r_iris.x * W
                l_x = l_iris.x * W
                iris = r_iris if abs(r_x - W/2) < abs(l_x - W/2) else l_iris
                cx = int(iris.x * W)
                cy = int(iris.y * H)
                return cx, cy
        except Exception as e:
            pass

    # HoughCircles
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    eyes = get_cascade().detectMultiScale(gray, 1.1, 5,
                                           minSize=(40,40), maxSize=(W//2,H//2))
    if len(eyes) > 0:
        ex,ey,ew,eh = sorted(eyes, key=lambda e:e[2]*e[3], reverse=True)[0]
        pad=10
        x1,y1=max(0,ex-pad),max(0,ey-pad)
        x2,y2=min(W,ex+ew+pad),min(H,ey+eh+pad)
    else:
        x1,y1=int(W*0.2),int(H*0.15)
        x2,y2=int(W*0.8),int(H*0.70)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    cw,ch = crop.shape[1], crop.shape[0]
    blur = cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(7,7),1.5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2,
                                min(cw,ch)//2, param1=50, param2=18,
                                minRadius=max(5,int(min(cw,ch)*0.08)),
                                maxRadius=max(20,int(min(cw,ch)*0.35)))
    if circles is None:
        return None, None

    circles = np.round(circles[0]).astype(int)
    eg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    best = min(circles, key=lambda c:
               cv2.mean(eg, cv2.circle(np.zeros(eg.shape,np.uint8),
                                       (c[0],c[1]),c[2],255,-1))[0])
    return x1+best[0], y1+best[1]

# Trained CNN from Kaggle dataset
def load_cnn(path, device):

    model = GazeCNN(img_size=128, dropout=0.5).to(device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"CNN loaded: {path}")
        model.eval()
        return model
    print(f"CNN not found ({path}) — using HoughCircles fallback")
    return None

def detect_pupil(frame_bgr, cnn_model=None, device='cpu'):
    """
      1. Haar cascade - ROI = eye
      2. GazeCNN -> pupil (x,y,r)
      3. convert to camera pixels
"""
    if frame_bgr is None:
        return None, None, None
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    #  Haar cascade — ROI
    eyes = get_cascade().detectMultiScale(gray, 1.1, 5,
                                           minSize=(40,40), maxSize=(W//2,H//2))
    if len(eyes) > 0:
        ex,ey,ew,eh = sorted(eyes, key=lambda e:e[2]*e[3], reverse=True)[0]
        pad=10
        x1,y1=max(0,ex-pad),max(0,ey-pad)
        x2,y2=min(W,ex+ew+pad),min(H,ey+eh+pad)
    else:
        x1,y1=int(W*0.2),int(H*0.15)
        x2,y2=int(W*0.8),int(H*0.70)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None
    cw, ch = crop.shape[1], crop.shape[0]

    # GazeCNN prediction
    if cnn_model is not None:
        try:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            t   = _CNN_TRANSFORM(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = cnn_model(t)[0]   # [x_norm, y_norm, r_norm] in [0,1]
            cx_cam = int(x1 + out[0].item() * cw)
            cy_cam = int(y1 + out[1].item() * ch)
            r_cam  = max(int(out[2].item() * min(cw,ch)), 5)
            return cx_cam, cy_cam, r_cam
        except Exception:
            pass  # fallback

    # HoughCircles fallback
    blur = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),(7,7),1.5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2,
                                min(cw,ch)//2, param1=50, param2=18,
                                minRadius=max(5,int(min(cw,ch)*0.08)),
                                maxRadius=max(20,int(min(cw,ch)*0.35)))
    if circles is None:
        return None, None, None
    circles = np.round(circles[0]).astype(int)
    eg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    best = min(circles, key=lambda c:
               cv2.mean(eg, cv2.circle(np.zeros(eg.shape,np.uint8),
                                       (c[0],c[1]),c[2],255,-1))[0])
    return x1+best[0], y1+best[1], best[2]


# Calibration - on board - 4 corners + 4 x edge middle + center
_BX = BOARD_X               # left side of the board
_BY = BOARD_Y               # top side of the board
_BW = BOARD_SIZE            # board width
_BH = BOARD_SIZE            # board height

CALIB_PTS = [
    (_BX + _BW//2,      _BY + _BH//2),       # center
    (_BX + 40,          _BY + 40),            # top left
    (_BX + _BW//2,      _BY + 40),            # top middle
    (_BX + _BW - 40,    _BY + 40),            # top right
    (_BX + 40,          _BY + _BH//2),        # left middle
    (_BX + _BW - 40,    _BY + _BH//2),        # right middle
    (_BX + 40,          _BY + _BH - 40),      # bottom left
    (_BX + _BW//2,      _BY + _BH - 40),      # bottom middle
    (_BX + _BW - 40,    _BY + _BH - 40),      # bottom right
]

def run_calibration(screen, cap, cnn_model=None, device='cpu'):
    screen_pts=[]; pupil_pts=[]; clock=pygame.time.Clock()

    for i,(sx,sy) in enumerate(CALIB_PTS):
        # wait for SPACE on each point
        while True:
            screen.fill(ui.BG)
            for j,(px,py) in enumerate(CALIB_PTS):
                if j<i:
                    pygame.draw.circle(screen,ui.GREEN_DARK,(px,py),8)
                    pygame.draw.circle(screen,ui.GREEN,(px,py),8,1)
                elif j==i:
                    r=14+int(4*np.sin(time.time()*4))
                    ui.glow_circle(screen,(px,py),r,ui.GREEN,3)
                    pygame.draw.circle(screen,ui.AMBER,(px,py),5)
                else:
                    pygame.draw.circle(screen,(60,66,80),(px,py),8,2)
            ui.text(screen,f"Calibration  {i+1}/{len(CALIB_PTS)}",
                    (WINDOW_W//2,WINDOW_H-76),18,ui.TEXT,bold=True,anchor='center')
            ui.text(screen,"Look at the dot, then press SPACE   ·   ESC to skip",
                    (WINDOW_W//2,WINDOW_H-46),13,ui.DIM,anchor='center')
            pygame.display.flip(); clock.tick(30)
            done=False
            for ev in pygame.event.get():
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_SPACE: done=True
                    elif ev.key==pygame.K_ESCAPE: return None
            if done: break

        # record
        samples=[]; collected=0
        while collected < 20:
            screen.fill(ui.BG)
            if cap:
                ret,frame=cap.read()
                if ret:
                    # HoughCircles during calibration
                    tx,ty=detect_pupil_tracking(frame)
                    if tx is not None:
                        cx,cy=tx,ty; r=10
                    else:
                        cx,cy,r=detect_pupil(frame,cnn_model,device)
                        if cx is None: continue
                    if cx is not None: samples.append((cx,cy)); collected+=1
                    rgb=cv2.cvtColor(cv2.resize(frame,(CAM_W,CAM_VID_H)),cv2.COLOR_BGR2RGB)
                    screen.blit(pygame.surfarray.make_surface(
                        np.transpose(rgb,(1,0,2))),(PANEL_X,BOARD_Y))
            ui.glow_circle(screen,(sx,sy),14,ui.GREEN,3)
            pygame.draw.circle(screen,ui.AMBER,(sx,sy),5)
            ui.text(screen,f"Recording...  {collected}/20",
                    (WINDOW_W//2,WINDOW_H-76),14,ui.TEXT,bold=True,anchor='center')
            ui.progress(screen,(WINDOW_W//2-150,WINDOW_H-52,300,12),collected/20)
            pygame.display.flip(); clock.tick(30)
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: return None

        if len(samples)>=5:
            arr=np.array(samples)
            screen_pts.append([float(sx),float(sy)])
            pupil_pts.append([float(np.median(arr[:,0])),float(np.median(arr[:,1]))])

    if len(screen_pts) < 4:
        return None

    src = np.array(pupil_pts,  dtype=np.float64)   # pupil coords
    dst = np.array(screen_pts, dtype=np.float64)   # screen coords

    def make_features(pts):
        f = np.column_stack([
            np.ones(len(pts)),
            pts[:,0], pts[:,1],
            pts[:,0]**2, pts[:,1]**2,
            pts[:,0]*pts[:,1]
        ])
        return f

    F = make_features(src)

    coef_x, _, _, _ = np.linalg.lstsq(F, dst[:,0], rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(F, dst[:,1], rcond=None)

    # calibration quality
    pred_x = F @ coef_x
    pred_y = F @ coef_y
    err_x = np.mean(np.abs(pred_x - dst[:,0]))
    err_y = np.mean(np.abs(pred_y - dst[:,1]))
    print(f"Calibration error: x={err_x:.1f}px  y={err_y:.1f}px")

    H = {'type': 'poly', 'coef_x': coef_x, 'coef_y': coef_y}
    np.save(CALIB_FILE, H, allow_pickle=True)
    return H


def pupil_to_screen(cx, cy, H):
    """
    H = dictionary
      - 'homography': cv2 homography matrix (original)
      - 'poly': polinomal coeffs (robust)
    """
    if H is None:
        return None, None

    # type = key dictionary
    if isinstance(H, dict):
        if H['type'] == 'poly':
            cx_f, cy_f = float(cx), float(cy)
            # screen_x = f(eye_x, eye_y)
            cx_f2, cy_f2 = cx_f**2, cy_f**2
            cxy = cx_f * cy_f
            features = np.array([1, cx_f, cy_f, cx_f2, cy_f2, cxy])
            gx = int(np.dot(H['coef_x'], features))
            gy = int(np.dot(H['coef_y'], features))
            gx = int(np.clip(gx, 0, WINDOW_W))
            gy = int(np.clip(gy, 0, WINDOW_H))
            return gx, gy

    # numpy matrix (homography)
    try:
        dst = cv2.perspectiveTransform(
            np.array([[[float(cx), float(cy)]]], np.float32), H)
        gx = int(np.clip(dst[0,0,0], 0, WINDOW_W))
        gy = int(np.clip(dst[0,0,1], 0, WINDOW_H))
        return gx, gy
    except Exception:
        return None, None

# Transformer

def load_transformer(path,device):
    if not os.path.exists(path): return None
    ckpt=torch.load(path,map_location=device); c=ckpt['config']
    m=build_transformer(source_context_size=c.get('seq_len',20),
                        target_context_size=2,
                        model_dimension=c.get('model_dimension',64),
                        number_of_blocks=c.get('number_of_blocks',2),
                        heads=c.get('heads',4), dropout=0.0,
                        feed_forward_dimension=c.get('feed_forward_dimension',256)
                        ).to(device)
    m.load_state_dict(ckpt['model_state_dict']); m.eval()
    print("Transformer loaded."); return m


def predict_probs(transformer,gaze_seq,device):
    if transformer is None or len(gaze_seq)<1:
        return np.ones(64)/64
    seq=list(gaze_seq)
    if len(seq)>TRIM: seq=seq[:-TRIM]
    if not seq: return np.ones(64)/64
    tokens=encode_sequence(','.join(seq),SEQ_LEN)
    src=torch.tensor(tokens,dtype=torch.long).unsqueeze(0).to(device)
    tgt=torch.full((1,1),SOS_IDX,dtype=torch.long,device=device)
    sm=(src!=PAD_IDX).unsqueeze(1).unsqueeze(1).int()
    tm=causal_mask(1).to(device)
    with torch.no_grad():
        e=transformer.encode(src,sm)
        d=transformer.decode(e,sm,tgt,tm)
        p=torch.softmax(transformer.project(d[:,-1])[0],dim=-1)
    return p.cpu().numpy()


# Puzzles

def load_puzzles(csv_path='data/lichess_puzzles.csv', n=50):
    import csv, random
    if not os.path.exists(csv_path):
        return FALLBACK_PUZZLES
    puzzles=[]
    with open(csv_path,'r',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        rows=[r for r in reader
              if 1000<=int(r.get('Rating',0))<=1600]
    if not rows: return FALLBACK_PUZZLES
    random.shuffle(rows)
    for row in rows[:n*3]:
        try:
            moves=row['Moves'].strip().split()
            if len(moves)<2: continue
            b=chess.Board(row['FEN'])
            mv=chess.Move.from_uci(moves[0])
            if mv not in b.legal_moves: continue
            b.push(mv)
            themes=row.get('Themes','').strip()
            desc=themes.split()[0] if themes else f"Rating {row.get('Rating','?')}"
            puzzles.append({'fen':b.fen(),'solution':moves[1],'desc':desc})
            if len(puzzles)>=n: break
        except Exception: continue
    return puzzles if puzzles else FALLBACK_PUZZLES


# Drawing functions

def sq_cr(sq, flipped=False):
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    if flipped:
        return 7-f, r
    return f, 7-r


def draw_piece(screen, fnt, sym, center, is_white):
    sh = fnt.render(sym, True, (20, 18, 16))
    screen.blit(sh, sh.get_rect(center=(center[0]+2, center[1]+2)))
    clr = (250, 250, 250) if is_white else (28, 28, 30)
    t = fnt.render(sym, True, clr)
    screen.blit(t, t.get_rect(center=center))


def draw_board_coords(screen, bx, by, sq_s, flipped):
    """Coordinates rendered inside the edge squares (lichess style)."""
    for i in range(8):
        fn = chess.FILE_NAMES[7-i] if flipped else chess.FILE_NAMES[i]
        clr = ui.BOARD_DARK if (i+7) % 2 == 0 else ui.BOARD_LIGHT
        ui.text(screen, fn, (bx+i*sq_s+sq_s-4, by+8*sq_s-2), 11, clr,
                bold=True, anchor='bottomright')
        rn = str(i+1) if flipped else str(8-i)
        clr = ui.BOARD_DARK if i % 2 == 0 else ui.BOARD_LIGHT
        ui.text(screen, rn, (bx+3, by+i*sq_s+1), 11, clr, bold=True)


def draw_board(screen,board,fp,probs,trail,gaze_sq,sel_sq=None,legal_sqs=None,flipped=False):
    # frame behind the board
    pygame.draw.rect(screen, ui.BOARD_FRAME,
                     (BOARD_X-6, BOARD_Y-6, BOARD_SIZE+12, BOARD_SIZE+12),
                     border_radius=8)
    top1=int(np.argmax(probs)) if probs is not None else None
    for sq in chess.SQUARES:
        col,row=sq_cr(sq,flipped)
        rect=pygame.Rect(BOARD_X+col*SQUARE_SIZE,
                         BOARD_Y+row*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE)
        pygame.draw.rect(screen,ui.BOARD_LIGHT if (col+row)%2==0 else ui.BOARD_DARK,rect)

        if probs is not None:
            p=probs[sq]
            if p>0.01:
                a=min(int(p*700),190)
                ui.alpha_rect(screen,rect,(*ui.BLUE,a))
                if p>0.06:
                    ui.text_shadow(screen,f'{p*100:.0f}%',(rect.x+4,rect.y+3),
                                   11,(255,255,255),bold=True)

        if top1 is not None and sq==top1:
            pygame.draw.rect(screen,ui.RED,rect,3,border_radius=4)

        if sel_sq is not None and sq==sel_sq:
            ui.alpha_rect(screen,rect,(20,200,20,110))

        if legal_sqs and sq in legal_sqs:
            s=pygame.Surface((SQUARE_SIZE,SQUARE_SIZE),pygame.SRCALPHA)
            pygame.draw.circle(s,(30,120,60,150),(SQUARE_SIZE//2,SQUARE_SIZE//2),9)
            screen.blit(s,rect.topleft)

    for i,name in enumerate(trail):
        try:
            tsq=chess.parse_square(name); tc,tr=sq_cr(tsq,flipped)
            a=int(30+160*(i/max(len(trail),1)))
            ui.alpha_rect(screen,(BOARD_X+tc*SQUARE_SIZE,BOARD_Y+tr*SQUARE_SIZE,
                                  SQUARE_SIZE,SQUARE_SIZE),(*ui.ORANGE,a))
        except: pass

    if gaze_sq is not None:
        try:
            gc,gr=sq_cr(gaze_sq,flipped)
            cx=BOARD_X+gc*SQUARE_SIZE+SQUARE_SIZE//2
            cy=BOARD_Y+gr*SQUARE_SIZE+SQUARE_SIZE//2
            ui.glow_circle(screen,(cx,cy),14,ui.GREEN,3)
        except: pass

    for sq in chess.SQUARES:
        p=board.piece_at(sq)
        if not p: continue
        sym=PIECES[p.piece_type][p.color]; col,row=sq_cr(sq,flipped)
        cx=BOARD_X+col*SQUARE_SIZE+SQUARE_SIZE//2
        cy=BOARD_Y+row*SQUARE_SIZE+SQUARE_SIZE//2
        draw_piece(screen,fp,sym,(cx,cy),p.color==chess.WHITE)

    draw_board_coords(screen,BOARD_X,BOARD_Y,SQUARE_SIZE,flipped)


def draw_camera(screen,frame_bgr,cx_cam,cy_cam,r_cam):
    rect = ui.card(screen,(PANEL_X,BOARD_Y,CAM_CARD_W,CAM_CARD_H),
                   title="Pupil detection")
    vx,vy = rect.x+12, rect.y+30
    if frame_bgr is not None:
        vis=frame_bgr.copy()
        if cx_cam is not None:
            cv2.circle(vis,(cx_cam,cy_cam),r_cam,(0,220,0),2)
            cv2.circle(vis,(cx_cam,cy_cam),r_cam//2,(0,160,0),1)
            cv2.line(vis,(cx_cam-12,cy_cam),(cx_cam+12,cy_cam),(255,255,255),1)
            cv2.line(vis,(cx_cam,cy_cam-12),(cx_cam,cy_cam+12),(255,255,255),1)
            cv2.circle(vis,(cx_cam,cy_cam),3,(0,0,255),-1)
        rgb=cv2.cvtColor(cv2.resize(vis,(CAM_W,CAM_VID_H)),cv2.COLOR_BGR2RGB)
        screen.blit(pygame.surfarray.make_surface(
            np.transpose(rgb,(1,0,2))),(vx,vy))
        pygame.draw.rect(screen,ui.CARD_BORDER,(vx,vy,CAM_W,CAM_VID_H),1)
        if cx_cam is not None:
            ui.pill(screen,"PUPIL DETECTED",(vx+6,vy+6),ui.GREEN_DARK,(190,255,205))
            ui.text(screen,f"cam ({cx_cam}, {cy_cam})   r = {r_cam}",
                    (vx,vy+CAM_VID_H+8),11,ui.DIM)
        else:
            ui.pill(screen,"SEARCHING...",(vx+6,vy+6),ui.RED_DARK,(255,190,190))
    else:
        pygame.draw.rect(screen,ui.CARD_INNER,(vx,vy,CAM_W,CAM_VID_H),border_radius=6)
        ui.text(screen,"No camera",(vx+CAM_W//2,vy+CAM_VID_H//2),12,ui.FAINT,
                anchor='center')
    ui.text(screen,"MediaPipe · CNN · Haar",(rect.right-12,rect.y+10),10,ui.FAINT,
            anchor='topright')


def draw_bar(screen,probs):
    rect = ui.card(screen,(PANEL_X,CHART_Y,PANEL_W,CHART_H),
                   title="Prediction — Top 10")
    x0=rect.x+12; y0=rect.y+34
    if probs is None or probs.max()<0.001:
        ui.text(screen,"Look at the board...",(rect.centerx,rect.centery),
                13,ui.FAINT,anchor='center')
        return
    top10=np.argsort(probs)[::-1][:10]; maxp=probs[top10[0]]
    lw=34; pct_w=52
    bmax=rect.width-24-lw-pct_w
    row_h=(rect.height-44)//10
    bh=min(row_h-6,14)
    for i,sq_idx in enumerate(top10):
        p=probs[sq_idx]; nm=idx_to_square(sq_idx)
        bw=int((p/max(maxp,1e-6))*bmax); yy=y0+i*row_h
        ui.text(screen,nm,(x0,yy+(bh-16)//2),12,
                ui.TEXT if i==0 else ui.DIM,bold=(i==0))
        pygame.draw.rect(screen,(46,51,64),(x0+lw,yy,bmax,bh),
                         border_radius=bh//2)
        if bw>=bh:
            pygame.draw.rect(screen,ui.RED if i==0 else ui.BLUE,
                             (x0+lw,yy,bw,bh),border_radius=bh//2)
        ui.text(screen,f"{p*100:.1f}%",(x0+lw+bmax+pct_w-4,yy+(bh-16)//2),11,
                ui.TEXT if i==0 else ui.DIM,anchor='topright')


def draw_result_overlay(screen, board, probs, move_san,
                        played_sq, predicted_sq, fp,
                        flipped=False):
    """
    Prediction overlay shown after a move is played:
    blue heatmap = model probabilities, green frame = move played,
    red frame = top-1 prediction, bar chart with top 10 predictions.
    """
    ui.alpha_rect(screen,(0,0,WINDOW_W,WINDOW_H),(8,9,12,200))

    PAD = 20
    px, py = PAD, PAD
    pw = WINDOW_W - 2*PAD
    ph = WINDOW_H - 2*PAD
    pygame.draw.rect(screen,ui.CARD,(px,py,pw,ph),border_radius=14)
    pygame.draw.rect(screen,ui.CARD_BORDER,(px,py,pw,ph),1,border_radius=14)

    # Title row
    played_name    = chess.square_name(played_sq)
    predicted_name = idx_to_square(predicted_sq)
    hit = (played_sq == predicted_sq)
    tr = ui.text(screen,f"Move: {move_san}   ·   Predicted: {predicted_name}",
                 (WINDOW_W//2, py+24),18,ui.TEXT,bold=True,anchor='center')
    if hit:
        ui.pill(screen,"HIT",(tr.right+14,tr.centery),ui.GREEN_DARK,
                (190,255,205),anchor='midleft')
    else:
        ui.pill(screen,"MISS",(tr.right+14,tr.centery),ui.RED_DARK,
                (255,190,190),anchor='midleft')

    # Legend
    leg_y = py + 48
    pygame.draw.rect(screen, ui.GREEN, (px+16, leg_y, 12, 12), 3, border_radius=3)
    ui.text(screen,f"played ({played_name})",(px+34,leg_y-2),12,ui.DIM)
    if predicted_sq != played_sq:
        pygame.draw.rect(screen, ui.RED, (px+180, leg_y, 12, 12), 3, border_radius=3)
        ui.text(screen,f"predicted ({predicted_name})",(px+198,leg_y-2),12,ui.DIM)
    hx = px+pw-220
    ui.hints(screen,(hx,leg_y-3),[("SPACE","next puzzle")])

    content_y  = leg_y + 24
    content_h  = ph - (content_y - py) - 14

    sq_s       = min(content_h // 8, (pw - 40) // 12)
    board_px   = sq_s * 8
    bx         = px + 16
    by         = content_y

    # Heatmap
    for sq in chess.SQUARES:
        col, row = sq_cr(sq, flipped=flipped) # keep the orientation
        rect = pygame.Rect(bx+col*sq_s, by+row*sq_s, sq_s, sq_s)
        pygame.draw.rect(screen, ui.BOARD_LIGHT if (col+row)%2==0 else ui.BOARD_DARK, rect)

        if probs is not None:
            p = probs[sq]
            if p > 0.008:
                a = min(int(p*800), 200)
                ui.alpha_rect(screen,rect,(*ui.BLUE,a))
                if p > 0.04:
                    ui.text_shadow(screen,f'{p*100:.0f}%',(rect.x+3,rect.y+2),
                                   10,(255,255,255),bold=True)
        if sq == played_sq:
            pygame.draw.rect(screen, ui.GREEN, rect, 4, border_radius=4)
        if sq == predicted_sq and predicted_sq != played_sq:
            pygame.draw.rect(screen, ui.RED, rect, 3, border_radius=4)

    font_mini = ui.piece_font(max(sq_s-6,12))
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p: continue
        sym = PIECES[p.piece_type][p.color]
        col, row = sq_cr(sq, flipped=flipped)
        cx = bx+col*sq_s+sq_s//2; cy = by+row*sq_s+sq_s//2
        draw_piece(screen,font_mini,sym,(cx,cy),p.color==chess.WHITE)

    # Board coordinates
    for i in range(8):
        fn = chess.FILE_NAMES[7-i] if flipped else chess.FILE_NAMES[i]
        ui.text(screen,fn,(bx+i*sq_s+sq_s//2, by+board_px+4),10,ui.FAINT,
                anchor='midtop')

    # Bar chart
    if probs is not None:
        bx2     = bx + board_px + 28
        by2     = by + 20
        bar_w   = px + pw - bx2 - 20
        lbl_w   = 34
        pct_w   = 48
        bar_max = bar_w - lbl_w - pct_w - 8
        row_h   = (content_h - 24) // 10
        bh      = min(max(row_h - 6, 8), 14)

        ui.text(screen,"TOP 10 PREDICTIONS",(bx2,by-2),11,ui.DIM,bold=True)

        top10 = np.argsort(probs)[::-1][:10]
        max_p = probs[top10[0]]

        for i, sq_idx in enumerate(top10):
            p   = probs[sq_idx]
            nm  = idx_to_square(sq_idx)
            bw  = int((p/max(max_p,1e-6))*bar_max)
            yy  = by2 + i*row_h

            if sq_idx == played_sq:    col_bar = ui.GREEN
            elif i == 0:               col_bar = ui.RED
            else:                      col_bar = ui.BLUE

            ui.text(screen,nm,(bx2,yy+(bh-16)//2),12,
                    ui.TEXT if i==0 or sq_idx==played_sq else ui.DIM,
                    bold=(i==0 or sq_idx==played_sq))
            pygame.draw.rect(screen,(46,51,64),(bx2+lbl_w,yy,bar_max,bh),
                             border_radius=bh//2)
            if bw>=bh:
                pygame.draw.rect(screen,col_bar,(bx2+lbl_w,yy,bw,bh),
                                 border_radius=bh//2)
            ui.text(screen,f"{p*100:.1f}%",(bx2+lbl_w+bar_max+pct_w,yy+(bh-16)//2),
                    11,ui.DIM,anchor='topright')


def draw_info_panel(screen, puzzle_no, n_puzzles, desc, mode_et, gaze_seq,
                    top1_nm):
    rect = ui.card(screen,(INFO_X,BOARD_Y,INFO_W,CAM_CARD_H),title="Session")
    x = rect.x+12
    ui.text(screen,f"Puzzle {puzzle_no}/{n_puzzles}",(x,rect.y+28),15,
            ui.TEXT,bold=True)
    ui.text(screen,desc[:26],(x,rect.y+52),12,ui.DIM)
    if mode_et:
        ui.pill(screen,"EYE TRACKER",(x,rect.y+74),ui.GREEN_DARK,(190,255,205))
    else:
        ui.pill(screen,"MOUSE MODE",(x,rect.y+74),ui.BLUE_DARK,(180,205,255))

    ui.text(screen,"GAZE SEQUENCE",(x,rect.y+102),10,ui.FAINT,bold=True)
    seq_fields = list(gaze_seq)[-6:]
    if seq_fields:
        ui.chips(screen,seq_fields,(x,rect.y+118),INFO_W-24)
    else:
        ui.text(screen,"—",(x,rect.y+118),12,ui.FAINT)

    pygame.draw.line(screen,ui.CARD_BORDER,(x,rect.y+166),
                     (rect.right-12,rect.y+166))
    ui.text(screen,"Top-1 prediction",(x,rect.y+178),11,ui.DIM)
    ui.text(screen,top1_nm,(rect.right-14,rect.y+172),18,ui.GREEN,
            bold=True,anchor='topright')


def draw_mode_select(screen, has_calib):
    screen.fill(ui.BG)
    ui.header(screen,WINDOW_W,"Eye-Tracking Chess",
              "Transformer-based move prediction from gaze")
    cw, ch = 480, 250
    rect = ui.card(screen,((WINDOW_W-cw)//2,(WINDOW_H-ch)//2-20,cw,ch),
                   radius=14)
    ui.text(screen,"Choose input mode",(rect.centerx,rect.y+30),17,
            ui.TEXT,bold=True,anchor='center')
    rows = [
        ("M","Mouse mode","recommended — mouse acts as gaze proxy",True),
        ("L","Load saved calibration",
         "use previous 9-point calibration" if has_calib else "no saved calibration found",
         has_calib),
        ("N","New calibration","9-point eye-tracking calibration",True),
    ]
    y = rect.y+64
    for key,label,sub,enabled in rows:
        kr = ui.keycap(screen,key,(rect.x+28,y+4),13)
        ui.text(screen,label,(rect.x+70,y),14,
                ui.TEXT if enabled else ui.FAINT,bold=True)
        ui.text(screen,sub,(rect.x+70,y+20),11,
                ui.DIM if enabled else ui.FAINT)
        y += 56
    ui.text(screen,"ESC also starts mouse mode",(rect.centerx,rect.bottom+24),
            11,ui.FAINT,anchor='center')
    pygame.display.flip()


def run():
    pygame.init()
    screen=pygame.display.set_mode((WINDOW_W,WINDOW_H))
    pygame.display.set_caption("Eye-Tracking Chess — Puzzle Demo")

    fp=ui.piece_font(SQUARE_SIZE-10)

    device=torch.device('cpu')
    transformer=load_transformer(TRANSFORMER_PATH,device)
    cnn_model=load_cnn(CNN_PATH,device)

    # Camera
    cap=None
    for idx in range(3):
        c=cv2.VideoCapture(idx)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap=c; print(f"Camera {idx}."); break

    # Calibration
    H_calib=None; use_et=(cap is not None)
    if use_et:
        has_calib = os.path.exists(CALIB_FILE)
        # Default: mouse; eye tracking kept as an option
        draw_mode_select(screen, has_calib)
        waiting=True
        while waiting:
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT:
                    if cap: cap.release()
                    pygame.quit(); return
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_n:
                        H_calib=run_calibration(screen,cap,cnn_model,device); waiting=False
                    elif ev.key==pygame.K_l and has_calib:
                        loaded = np.load(CALIB_FILE, allow_pickle=True)
                        H_calib = loaded.item() if loaded.ndim == 0 else loaded
                        print("Calibration loaded."); waiting=False
                    elif ev.key in (pygame.K_m, pygame.K_ESCAPE):
                        use_et=False; waiting=False

    # Puzzles
    puzzles=load_puzzles(); puzzle_idx=0

    def load_puzzle(idx):
        p=puzzles[idx%len(puzzles)]
        return chess.Board(p['fen']), p['solution'], p['desc']

    board,solution_uci,desc=load_puzzle(puzzle_idx)
    solution_move=chess.Move.from_uci(solution_uci)
    solution_sq  =solution_move.to_square
    player_color =board.turn

    # Status
    gaze_seq  =deque(maxlen=GAZE_WINDOW*3)
    logged_sq =None; dwell=0
    probs     =np.ones(64)/64
    gaze_sq   =None
    sel_sq    =None
    frame_cnt =0
    cx_cam=cy_cam=r_cam=None
    frame_bgr =None
    gaze_filt =GazeFilter()
    clock     =pygame.time.Clock()

    # Overlay status
    show_overlay   =False
    overlay_probs  =None
    board_before   =None
    move_san_str   =""
    predicted_sq_ov=None
    played_sq_ov   =None
    overlay_flipped=False  # board orientation at the moment of the move

    print("Demo: solve puzzles! ESC=exit  K=calibrate  SPACE(overlay)=next")

    while True:
        frame_cnt+=1
        mx,my=pygame.mouse.get_pos()

        # Camera + pupil detction
        if cap is not None:
            ret,frame_bgr=cap.read()
            if ret:
                # CNN — visualization - green circle
                cx_cam,cy_cam,r_cam=detect_pupil(frame_bgr,cnn_model,device)
                # HoughCircles — tracking + calibration
                tx,ty=detect_pupil_tracking(frame_bgr)
                if tx is not None:
                    gaze_filt.update(tx,ty)
                elif cx_cam is not None:
                    gaze_filt.update(cx_cam,cy_cam)
            else:
                cx_cam=cy_cam=r_cam=None

        # Track eye coordinatates on screen
        if use_et and H_calib is not None:
            sx,sy=gaze_filt.get()
            if sx is not None:
                gx,gy=pupil_to_screen(sx,sy,H_calib)
            else:
                gx,gy=mx,my
        else:
            gx,gy=mx,my

        # Flip
        flipped = (board.turn == chess.BLACK)

        # map to square
        res=screen_to_square(gx,gy,
                              board_x=BOARD_X,board_y=BOARD_Y,
                              board_w=BOARD_SIZE,board_h=BOARD_SIZE,
                              flipped=flipped)
        if res['valid']:
            try:    sq=chess.parse_square(res['square'])
            except: sq=None
        else: sq=None

        gaze_sq=sq

        # add fixation
        if sq is not None:
            if sq==logged_sq:
                dwell+=1
            else:
                dwell=0; logged_sq=sq
            if dwell==MIN_DWELL:
                gaze_seq.append(chess.square_name(sq))

        # transformer predictions (updating all the time)
        if frame_cnt%UPDATE_EVERY==0 and not show_overlay:
            probs=predict_probs(transformer,gaze_seq,device)

        # legal moves for selected piece
        legal_sqs=set()
        if sel_sq is not None:
            for mv in board.legal_moves:
                if mv.from_square==sel_sq:
                    legal_sqs.add(mv.to_square)

        # key events + mouse events
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                if cap: cap.release()
                pygame.quit(); return

            elif ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE:
                    if cap: cap.release()
                    pygame.quit(); return
                elif ev.key==pygame.K_k and cap:
                    H_calib=run_calibration(screen,cap,cnn_model,device)
                    use_et=(H_calib is not None)
                    gaze_filt.reset()
                elif ev.key==pygame.K_SPACE and show_overlay:
                    # Next puzzle
                    puzzle_idx+=1
                    board,solution_uci,desc=load_puzzle(puzzle_idx)
                    solution_move=chess.Move.from_uci(solution_uci)
                    solution_sq  =solution_move.to_square
                    player_color =board.turn
                    gaze_seq.clear(); probs=np.ones(64)/64
                    sel_sq=None; show_overlay=False; gaze_filt.reset()

            elif ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1 and not show_overlay:
                # Click on the board — play a move
                res2=screen_to_square(mx,my,
                                       board_x=BOARD_X,board_y=BOARD_Y,
                                       board_w=BOARD_SIZE,board_h=BOARD_SIZE,
                                       flipped=flipped)
                if not res2['valid']:
                    sel_sq=None; continue
                try:    clicked=chess.parse_square(res2['square'])
                except: continue

                if sel_sq is None:
                    p=board.piece_at(clicked)
                    if p and p.color==player_color:
                        sel_sq=clicked
                else:
                    mv=chess.Move(sel_sq,clicked)
                    # promotion
                    p=board.piece_at(sel_sq)
                    if p and p.piece_type==chess.PAWN and chess.square_rank(clicked) in (0,7):
                        mv=chess.Move(sel_sq,clicked,promotion=chess.QUEEN)
                    if mv in board.legal_moves:
                        # board fen before the move has been played
                        board_before  =board.copy()
                        overlay_probs =probs.copy()
                        predicted_sq_ov=int(np.argmax(probs))
                        move_san_str  =board.san(mv)
                        overlay_correct=(clicked==solution_sq)
                        overlay_flipped=flipped  # orientation
                        board.push(mv)
                        sel_sq=None
                        show_overlay=True
                        played_sq_ov=clicked   # played square
                    else:
                        p=board.piece_at(clicked)
                        if p and p.color==player_color: sel_sq=clicked
                        else: sel_sq=None


        screen.fill(ui.BG)

        mode_et = use_et and H_calib is not None
        ui.header(screen,WINDOW_W,"Eye-Tracking Chess",
                  "Transformer-based move prediction from gaze",
                  right_pill=(("EYE TRACKER",ui.GREEN_DARK,(190,255,205))
                              if mode_et else
                              ("MOUSE MODE",ui.BLUE_DARK,(180,205,255))))

        draw_board(screen,board,fp,
                   None if show_overlay else probs,
                   gaze_seq,gaze_sq,sel_sq,legal_sqs,flipped=flipped)

        draw_camera(screen,frame_bgr,cx_cam,cy_cam,r_cam)
        draw_bar(screen,None if show_overlay else probs)

        # gaze crosshair
        if not show_overlay and 0<=gx<WINDOW_W and 0<=gy<WINDOW_H:
            ui.glow_circle(screen,(gx,gy),14,ui.AMBER,2)
            pygame.draw.line(screen,ui.AMBER,(gx-20,gy),(gx-8,gy),2)
            pygame.draw.line(screen,ui.AMBER,(gx+8,gy),(gx+20,gy),2)
            pygame.draw.line(screen,ui.AMBER,(gx,gy-20),(gx,gy-8),2)
            pygame.draw.line(screen,ui.AMBER,(gx,gy+8),(gx,gy+20),2)
            pygame.draw.circle(screen,ui.AMBER,(gx,gy),3)

        top1_nm = idx_to_square(int(np.argmax(probs))) if len(gaze_seq)>=1 else "..."
        puzzle_no = puzzle_idx % len(puzzles) + 1

        draw_info_panel(screen,puzzle_no,len(puzzles),desc,mode_et,
                        gaze_seq,top1_nm)

        # keyboard hints under the board (the overlay has its own hint)
        hint_y = BOARD_Y+BOARD_SIZE+16
        if not show_overlay:
            ui.hints(screen,(BOARD_X,hint_y),[("K","calibrate"),("ESC","quit")])

        # pupil debug readout (eye-tracking mode)
        sx_dbg,sy_dbg = gaze_filt.get() if use_et else (None,None)
        if show_overlay:
            pass
        elif sx_dbg is not None and H_calib is not None:
            gx_d,gy_d = pupil_to_screen(sx_dbg,sy_dbg,H_calib)
            ui.text(screen,f"pupil ({sx_dbg}, {sy_dbg})  →  screen ({gx_d}, {gy_d})",
                    (BOARD_X+BOARD_SIZE,hint_y+3),11,ui.FAINT,anchor='topright')
        elif sx_dbg is not None:
            ui.text(screen,f"pupil ({sx_dbg}, {sy_dbg})  —  not calibrated",
                    (BOARD_X+BOARD_SIZE,hint_y+3),11,ui.RED,anchor='topright')

        # Overlay after the move
        if show_overlay and board_before is not None:
            draw_result_overlay(screen,board_before,overlay_probs,
                                move_san_str,
                                played_sq_ov,predicted_sq_ov,fp,
                                flipped=overlay_flipped)

        pygame.display.flip()
        clock.tick(30)


if __name__=='__main__':
    run()
