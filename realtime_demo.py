"""
- Real-time eye-tracking/mouse mode + puzzle mode with heatmaps
- With calibration
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

# Layout
BOARD_OFFSET = 40
BOARD_SIZE   = 560
SQUARE_SIZE  = BOARD_SIZE // 8


# Dimensions
PANEL_X   = BOARD_OFFSET + BOARD_SIZE + 20
CAM_W     = 210   
CAM_H     = 180    

GAP       = 10     
INFO_W    = 210   
TOP_H     = CAM_H  

PANEL_W   = CAM_W + GAP + INFO_W          
LBL_H     = 18    
BAR_Y     = BOARD_OFFSET + TOP_H + LBL_H + 8  
BAR_H     = BOARD_SIZE - (BAR_Y - BOARD_OFFSET) - 4

WINDOW_W  = BOARD_OFFSET + BOARD_SIZE + 20 + PANEL_W + 20
WINDOW_H  = BOARD_OFFSET + BOARD_SIZE + BOARD_OFFSET

# Y coordinates
CAM_Y     = BOARD_OFFSET                  
INFO_Y    = BOARD_OFFSET                 
CAM_LBL_Y = BOARD_OFFSET + TOP_H + 2     
BAR_LBL_Y = CAM_LBL_Y + LBL_H + 2      
INFO_X    = PANEL_X + CAM_W + GAP        

CAM_H_DISPLAY = CAM_H

TRANSFORMER_PATH = 'checkpoints/best_transformer.pth'
CNN_PATH         = 'checkpoints/best_model.pth'
CALIB_FILE       = 'checkpoints/calibration.npy'

GAZE_WINDOW  = 20
TRIM         = 3
MIN_DWELL    = 8      # frames including the same filed - fixation
UPDATE_EVERY = 10     # frames between predictions (faster update)
SMOOTH_N     = 25     # moving average pupil coords (more -> more stable)

C_LIGHT=(240,217,181); C_DARK=(181,136,99)
C_BG=(30,30,30);       C_PANEL=(45,45,45)
C_TEXT=(220,220,220);  C_DIM=(140,140,140)
C_GREEN=(50,210,50);   C_RED=(220,60,60)
C_ORANGE=(255,140,40); C_YELLOW=(240,200,0)

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
    {'fen':'4k3/8/4K3/4R3/8/8/8/8 w - - 0 1','solution':'e5e8','desc':'Mat u 1 — top na e8'},
    {'fen':'6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1','solution':'e1e8','desc':'Top na 8. redu'},
    {'fen':'r3k2r/ppp2ppp/2n5/3p4/3P4/2N5/PPP2PPP/R4RK1 w kq - 0 1','solution':'f1f8','desc':'Napad topom'},
    {'fen':'8/8/8/3k4/8/3K4/8/4R3 w - - 0 1','solution':'e1e5','desc':'Top na 5. redu'},
    {'fen':'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
     'solution':'f3g5','desc':'Napad na f7'},
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
        print(f"CNN ucitan: {path}")
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
                out = cnn_model(t)[0]   # [x_norm, y_norm, r_norm] u [0,1]
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
_BX = BOARD_OFFSET          # left side of the board
_BY = BOARD_OFFSET          # top side of the board
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

def run_calibration(screen, cap, fb, fu, cnn_model=None, device='cpu'):
    screen_pts=[]; pupil_pts=[]; clock=pygame.time.Clock()

    for i,(sx,sy) in enumerate(CALIB_PTS):
        # space for calibrate
        while True:
            screen.fill((20,20,20))
            for j,(px,py) in enumerate(CALIB_PTS):
                if j<i:    pygame.draw.circle(screen,(0,120,0),(px,py),8)
                elif j==i:
                    r=14+int(4*np.sin(time.time()*4))
                    pygame.draw.circle(screen,C_GREEN,(px,py),r)
                    pygame.draw.circle(screen,C_YELLOW,(px,py),5)
                else:      pygame.draw.circle(screen,C_DIM,(px,py),8)
            screen.blit(fb.render(f"Kalibracija  {i+1}/{len(CALIB_PTS)}",True,C_TEXT),
                        fb.render(f"Kalibracija  {i+1}/{len(CALIB_PTS)}",True,C_TEXT
                                  ).get_rect(center=(WINDOW_W//2,WINDOW_H-70)))
            screen.blit(fu.render("Gledaj tačku → SPACE   |   ESC=preskoci",True,C_DIM),
                        fu.render("Gledaj tačku → SPACE   |   ESC=preskoci",True,C_DIM
                                  ).get_rect(center=(WINDOW_W//2,WINDOW_H-40)))
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
            screen.fill((20,20,20))
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
                    rgb=cv2.cvtColor(cv2.resize(frame,(CAM_W,CAM_H)),cv2.COLOR_BGR2RGB)
                    screen.blit(pygame.surfarray.make_surface(
                        np.transpose(rgb,(1,0,2))),(PANEL_X,BOARD_OFFSET))
            pygame.draw.circle(screen,C_GREEN,(sx,sy),14)
            pygame.draw.circle(screen,C_YELLOW,(sx,sy),5)
            bw=int(300*collected/20)
            pygame.draw.rect(screen,(60,60,60),(WINDOW_W//2-150,WINDOW_H-50,300,16))
            pygame.draw.rect(screen,C_GREEN,(WINDOW_W//2-150,WINDOW_H-50,bw,16))
            screen.blit(fu.render(f"Snimam... {collected}/20",True,C_TEXT),
                        fu.render(f"Snimam... {collected}/20",True,C_TEXT
                                  ).get_rect(center=(WINDOW_W//2,WINDOW_H-70)))
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
    print(f"Kalibracija greska: x={err_x:.1f}px  y={err_y:.1f}px")

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
    print("Transformer učitan."); return m


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

def draw_board(screen,board,fp,fl,probs,trail,gaze_sq,sel_sq=None,legal_sqs=None,flipped=False):
    top1=int(np.argmax(probs)) if probs is not None else None
    for sq in chess.SQUARES:
        col,row=sq_cr(sq,flipped)
        rect=pygame.Rect(BOARD_OFFSET+col*SQUARE_SIZE,
                         BOARD_OFFSET+row*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE)
        pygame.draw.rect(screen,C_LIGHT if (col+row)%2==0 else C_DARK,rect)

        if probs is not None:
            p=probs[sq]
            if p>0.01:
                a=min(int(p*700),190)
                s=pygame.Surface((SQUARE_SIZE,SQUARE_SIZE),pygame.SRCALPHA)
                s.fill((30,100,220,a)); screen.blit(s,rect.topleft)
                if p>0.06:
                    screen.blit(fl.render(f'{p*100:.0f}%',True,(255,255,255)),
                                (rect.x+3,rect.y+3))

        if top1 is not None and sq==top1:
            pygame.draw.rect(screen,C_RED,rect,3)

        if sel_sq is not None and sq==sel_sq:
            s=pygame.Surface((SQUARE_SIZE,SQUARE_SIZE),pygame.SRCALPHA)
            s.fill((20,200,20,150)); screen.blit(s,rect.topleft)

        if legal_sqs and sq in legal_sqs:
            cx2=rect.x+SQUARE_SIZE//2; cy2=rect.y+SQUARE_SIZE//2
            pygame.draw.circle(screen,(100,200,100),(cx2,cy2),8)

    for i,name in enumerate(trail):
        try:
            tsq=chess.parse_square(name); tc,tr=sq_cr(tsq,flipped)
            a=int(30+160*(i/max(len(trail),1)))
            s=pygame.Surface((SQUARE_SIZE,SQUARE_SIZE),pygame.SRCALPHA)
            s.fill((255,140,40,a))
            screen.blit(s,(BOARD_OFFSET+tc*SQUARE_SIZE,BOARD_OFFSET+tr*SQUARE_SIZE))
        except: pass

    if gaze_sq is not None:
        try:
            gc,gr=sq_cr(gaze_sq,flipped)
            cx=BOARD_OFFSET+gc*SQUARE_SIZE+SQUARE_SIZE//2
            cy=BOARD_OFFSET+gr*SQUARE_SIZE+SQUARE_SIZE//2
            pygame.draw.circle(screen,C_GREEN,(cx,cy),14,3)
        except: pass

    for sq in chess.SQUARES:
        p=board.piece_at(sq)
        if not p: continue
        sym=PIECES[p.piece_type][p.color]; col,row=sq_cr(sq,flipped)
        cx=BOARD_OFFSET+col*SQUARE_SIZE+SQUARE_SIZE//2
        cy=BOARD_OFFSET+row*SQUARE_SIZE+SQUARE_SIZE//2
        screen.blit(fp.render(sym,True,(0,0,0)),
                    fp.render(sym,True,(0,0,0)).get_rect(center=(cx+2,cy+2)))
        clr=(255,255,255) if p.color==chess.WHITE else (25,25,25)
        screen.blit(fp.render(sym,True,clr),
                    fp.render(sym,True,clr).get_rect(center=(cx,cy)))

    for i in range(8):
        fn = chess.FILE_NAMES[7-i] if flipped else chess.FILE_NAMES[i]
        rn = str(i+1) if flipped else str(8-i)
        screen.blit(fl.render(fn,True,C_TEXT),
                    (BOARD_OFFSET+i*SQUARE_SIZE+3,BOARD_OFFSET+BOARD_SIZE+4))
        screen.blit(fl.render(rn,True,C_TEXT),
                    (BOARD_OFFSET-18,BOARD_OFFSET+i*SQUARE_SIZE+SQUARE_SIZE//2-8))
    pygame.draw.rect(screen,(70,70,70),
                     (BOARD_OFFSET,BOARD_OFFSET,BOARD_SIZE,BOARD_SIZE),2)


def draw_camera(screen,frame_bgr,cx_cam,cy_cam,r_cam,fu):
    x0,y0 = PANEL_X, CAM_Y
    pygame.draw.rect(screen,(70,70,70),(x0-2,y0-2,CAM_W+4,CAM_H+4),2)
    if frame_bgr is not None:
        vis=frame_bgr.copy()
        if cx_cam is not None:
            cv2.circle(vis,(cx_cam,cy_cam),r_cam,(0,220,0),2)
            cv2.circle(vis,(cx_cam,cy_cam),r_cam//2,(0,160,0),1)
            cv2.line(vis,(cx_cam-12,cy_cam),(cx_cam+12,cy_cam),(255,255,255),1)
            cv2.line(vis,(cx_cam,cy_cam-12),(cx_cam,cy_cam+12),(255,255,255),1)
            cv2.circle(vis,(cx_cam,cy_cam),3,(0,0,255),-1)
        rgb=cv2.cvtColor(cv2.resize(vis,(CAM_W,CAM_H)),cv2.COLOR_BGR2RGB)  # CAM_W x CAM_H
        screen.blit(pygame.surfarray.make_surface(
            np.transpose(rgb,(1,0,2))),(x0,y0))
        s=pygame.Surface((CAM_W,20),pygame.SRCALPHA)
        if cx_cam is not None:
            s.fill((0,130,0,170)); screen.blit(s,(x0,y0))
            screen.blit(fu.render("● ZENICA DETEKTOVANA",True,(200,255,200)),(x0+6,y0+3))
            screen.blit(fu.render(f"x={cx_cam}  y={cy_cam}  r={r_cam}",True,C_GREEN),
                        (x0+4,y0+CAM_H-18))
        else:
            s.fill((130,0,0,170)); screen.blit(s,(x0,y0))
            screen.blit(fu.render("○ Tražim zenicu...",True,(255,180,180)),(x0+6,y0+3))
    else:
        pygame.draw.rect(screen,(40,40,40),(x0,y0,CAM_W,CAM_H))
        screen.blit(fu.render("Nema kamere",True,C_DIM),(x0+10,y0+CAM_H//2))
    screen.blit(fu.render("Detekcija zenice (CNN + Haar)",True,C_DIM),(x0,CAM_LBL_Y))


def draw_bar(screen,probs,fu):
    x0=PANEL_X; y0=BAR_Y; w=PANEL_W; h=BAR_H
    pygame.draw.rect(screen,(38,38,38),(x0,y0,w,h))
    screen.blit(fu.render("Verovatnoce — Top 10",True,C_TEXT),(x0,BAR_LBL_Y))
    if probs is None or probs.max()<0.001:
        screen.blit(fu.render("Gledaj u tablu...",True,C_DIM),(x0+8,y0+h//2)); return
    top10=np.argsort(probs)[::-1][:10]; maxp=probs[top10[0]]
    lw=30; bmax=w-lw-52; bh=max(1,(h-14)//10-3)
    for i,sq_idx in enumerate(top10):
        p=probs[sq_idx]; nm=idx_to_square(sq_idx)
        bw=int((p/max(maxp,1e-6))*bmax); yy=y0+8+i*(bh+3)
        pygame.draw.rect(screen,C_RED if i==0 else (50,110,195),(x0+lw,yy,bw,bh))
        pygame.draw.rect(screen,(60,60,60),(x0+lw,yy,bmax,bh),1)
        screen.blit(fu.render(nm,True,C_TEXT),(x0+2,yy))
        screen.blit(fu.render(f"{p*100:.1f}%",True,C_DIM),(x0+lw+bmax+4,yy))


def draw_result_overlay(screen, board, probs, move_san,
                        played_sq, predicted_sq, fp, fl, fu, fb,
                        flipped=False):
    """
    Prediction overlay after the move is made
    # blue - probabilites
    # green frame - move played
    # red frame - move predicted (top 1)
    # bar chart with top 10 predictions
    """
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((0,0,0,190))
    screen.blit(overlay,(0,0))

   
    PAD = 20
    px, py = PAD, PAD
    pw = WINDOW_W - 2*PAD
    ph = WINDOW_H - 2*PAD
    pygame.draw.rect(screen,(28,28,28),(px,py,pw,ph),border_radius=10)
    pygame.draw.rect(screen,(80,80,80),(px,py,pw,ph),2,border_radius=10)

    # Top text
    played_name    = chess.square_name(played_sq)
    predicted_name = idx_to_square(predicted_sq)
    hit = (played_sq == predicted_sq)
    title = fb.render(
        f"Potez: {move_san}  |  Prediction: {predicted_name}",
        True, C_GREEN if hit else C_TEXT)
    screen.blit(title, title.get_rect(center=(WINDOW_W//2, py+22)))

    leg_y = py + 44
    pygame.draw.rect(screen, C_GREEN, (px+10, leg_y, 12, 12), 3)
    screen.blit(fu.render(f"= played ({played_name})", True, C_TEXT), (px+26, leg_y))
    if predicted_sq != played_sq:
        pygame.draw.rect(screen, C_RED, (px+200, leg_y, 12, 12), 3)
        screen.blit(fu.render(f"= prediction ({predicted_name})", True, C_TEXT), (px+216, leg_y))
    screen.blit(fu.render("SPACE = next puzzle", True, C_YELLOW),
                (pw - 160, leg_y))


    content_y  = leg_y + 20
    content_h  = ph - (content_y - py) - 10
   
    sq_s       = min(content_h // 8, (pw - 40) // 12) 
    board_px   = sq_s * 8
    bx         = px + 10
    by         = content_y

    # Heatmap
    for sq in chess.SQUARES:
        col, row = sq_cr(sq, flipped=flipped) # keep the orientation
        rect = pygame.Rect(bx+col*sq_s, by+row*sq_s, sq_s, sq_s)
        pygame.draw.rect(screen, C_LIGHT if (col+row)%2==0 else C_DARK, rect)

        if probs is not None:
            p = probs[sq]
            if p > 0.008:
                a = min(int(p*800), 200)
                s = pygame.Surface((sq_s,sq_s), pygame.SRCALPHA)
                s.fill((30,100,220,a)); screen.blit(s,rect.topleft)
                if p > 0.04:
                    screen.blit(fu.render(f'{p*100:.0f}%',True,(255,255,255)),
                                (rect.x+2,rect.y+2))
        if sq == played_sq:
            pygame.draw.rect(screen, C_GREEN, rect, 4)
        if sq == predicted_sq and predicted_sq != played_sq:
            pygame.draw.rect(screen, C_RED, rect, 3)

    font_mini = pygame.font.SysFont('segoeuisymbol', max(sq_s-6,12))
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p: continue
        sym = PIECES[p.piece_type][p.color]
        col, row = sq_cr(sq, flipped=flipped)
        cx = bx+col*sq_s+sq_s//2; cy = by+row*sq_s+sq_s//2
        clr = (255,255,255) if p.color==chess.WHITE else (25,25,25)
        screen.blit(font_mini.render(sym,True,(0,0,0)),
                    font_mini.render(sym,True,(0,0,0)).get_rect(center=(cx+1,cy+1)))
        screen.blit(font_mini.render(sym,True,clr),
                    font_mini.render(sym,True,clr).get_rect(center=(cx,cy)))

    # Board coordinates
    for i in range(8):
        screen.blit(fl.render(chess.FILE_NAMES[i],True,C_DIM),
                    (bx+i*sq_s+2, by+board_px+2))

    # Bar chart
    if probs is not None:
        bx2     = bx + board_px + 20
        by2     = by
        bar_w   = pw - board_px - 50      
        lbl_w   = 32
        pct_w   = 42
        bar_max = bar_w - lbl_w - pct_w - 8
        row_h   = content_h // 10
        bh      = max(row_h - 5, 8)

        screen.blit(fu.render("Top-10", True, C_TEXT), (bx2, by2-16))

        top10 = np.argsort(probs)[::-1][:10]
        max_p = probs[top10[0]]

        for i, sq_idx in enumerate(top10):
            p   = probs[sq_idx]
            nm  = idx_to_square(sq_idx)
            bw  = int((p/max(max_p,1e-6))*bar_max)
            yy  = by2 + i*row_h + 2

            if sq_idx == played_sq:    col_bar = C_GREEN
            elif i == 0:               col_bar = C_RED
            else:                      col_bar = (50,110,195)

            pygame.draw.rect(screen, col_bar,    (bx2+lbl_w, yy, bw, bh))
            pygame.draw.rect(screen, (60,60,60), (bx2+lbl_w, yy, bar_max, bh), 1)
            screen.blit(fu.render(nm,  True, C_TEXT), (bx2+2,              yy))
            screen.blit(fu.render(f"{p*100:.1f}%", True, C_DIM),
                        (bx2+lbl_w+bar_max+4, yy))






def run():
    pygame.init()
    screen=pygame.display.set_mode((WINDOW_W,WINDOW_H))
    pygame.display.set_caption("Eye-tracking Chess — Puzzle Demo")

    fp=pygame.font.SysFont('segoeuisymbol',SQUARE_SIZE-10)
    fl=pygame.font.SysFont('arial',13)
    fu=pygame.font.SysFont('arial',13)
    fb=pygame.font.SysFont('arial',20,bold=True)

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
        screen.fill((20,20,20))
        has_calib = os.path.exists(CALIB_FILE)
        # Default: mouse
        # keep eye tracking as an option (not rly working :( )
        options = [
            ("Eye-tracking Chess — Demo", True),
            ("", False),
            ("M =  Mouse mode  (recommended)", False),
            ("L  =  Load calibration" if has_calib else "  calibration doesn't exist", False),
            ("N  =  New Calibration (9 dots)", False),
        ]
        for i,(ln,bold) in enumerate(options):
            if not ln: continue
            f=fb if bold else fu
            col=C_TEXT if bold else (C_GREEN if i==2 else C_DIM)
            t=f.render(ln,True,col)
            screen.blit(t,t.get_rect(center=(WINDOW_W//2,WINDOW_H//2-60+i*34)))
        pygame.display.flip()
        waiting=True
        while waiting:
            for ev in pygame.event.get():
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_n:
                        H_calib=run_calibration(screen,cap,fb,fu,cnn_model,device); waiting=False
                    elif ev.key==pygame.K_l and has_calib:
                        loaded = np.load(CALIB_FILE, allow_pickle=True)
                        H_calib = loaded.item() if loaded.ndim == 0 else loaded
                        print("Kalibracija ucitana."); waiting=False
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
    overlay_flipped=False  # orijentacija table u trenutku poteza

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
                              board_x=BOARD_OFFSET,board_y=BOARD_OFFSET,
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
                    H_calib=run_calibration(screen,cap,fb,fu,cnn_model,device)
                    use_et=(H_calib is not None)
                    gaze_filt.reset()
                elif ev.key==pygame.K_SPACE and show_overlay:
                    # Sledeća puzla
                    puzzle_idx+=1
                    board,solution_uci,desc=load_puzzle(puzzle_idx)
                    solution_move=chess.Move.from_uci(solution_uci)
                    solution_sq  =solution_move.to_square
                    player_color =board.turn
                    gaze_seq.clear(); probs=np.ones(64)/64
                    sel_sq=None; show_overlay=False; gaze_filt.reset()

            elif ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1 and not show_overlay:
                # Klik na tablu — igranje poteza
                res2=screen_to_square(mx,my,
                                       board_x=BOARD_OFFSET,board_y=BOARD_OFFSET,
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


        screen.fill(C_BG)
        pygame.draw.rect(screen,C_PANEL,(PANEL_X-10,0,WINDOW_W-PANEL_X+10,WINDOW_H))

        draw_board(screen,board,fp,fl,
                   None if show_overlay else probs,
                   gaze_seq,gaze_sq,sel_sq,legal_sqs,flipped=flipped)

        draw_camera(screen,frame_bgr,cx_cam,cy_cam,r_cam,fu)
        draw_bar(screen,None if show_overlay else probs,fu)
        # + for gaze
        if not show_overlay and 0<=gx<WINDOW_W and 0<=gy<WINDOW_H:
            pygame.draw.circle(screen,C_YELLOW,(gx,gy),14,2)
            pygame.draw.line(screen,C_YELLOW,(gx-18,gy),(gx+18,gy),2)
            pygame.draw.line(screen,C_YELLOW,(gx,gy-18),(gx,gy+18),2)
            pygame.draw.circle(screen,C_YELLOW,(gx,gy),3)

    
        mode = "Eye-tracker" if use_et and H_calib is not None else "Mouse (proxy for gaze)"
        sx_dbg,sy_dbg = gaze_filt.get() if use_et else (None,None)
        top1_nm = idx_to_square(int(np.argmax(probs))) if len(gaze_seq)>=1 else "..."
        puzla_br = puzzle_idx % len(puzzles) + 1

        pygame.draw.rect(screen,(40,40,40),
                         (INFO_X, INFO_Y, INFO_W, TOP_H), border_radius=6)
        pygame.draw.rect(screen,(65,65,65),
                         (INFO_X, INFO_Y, INFO_W, TOP_H), 1, border_radius=6)

        pygame.draw.line(screen,(65,65,65),
                         (PANEL_X, BAR_Y-4),(PANEL_X+PANEL_W, BAR_Y-4),1)

        fb13 = pygame.font.SysFont('arial',13,bold=True)
        fu13 = pygame.font.SysFont('arial',13)

        y_i = INFO_Y + 8
  
        screen.blit(fb13.render(f"Puzzle {puzla_br}/{len(puzzles)}",True,C_TEXT),
                    (INFO_X+6, y_i)); y_i+=18
        
        # for debug
        if sx_dbg is not None and H_calib is not None:
            gx_d,gy_d = pupil_to_screen(sx_dbg,sy_dbg,H_calib)
            screen.blit(fu13.render(f"Pupil on camera: ({sx_dbg},{sy_dbg})",True,C_DIM),(INFO_X+6,y_i)); y_i+=15
            screen.blit(fu13.render(f"On screen:  ({gx_d},{gy_d})",True,C_YELLOW),(INFO_X+6,y_i)); y_i+=15
        elif sx_dbg is not None:
            screen.blit(fu13.render(f"Pupil: ({sx_dbg},{sy_dbg})",True,C_DIM),(INFO_X+6,y_i)); y_i+=15
            screen.blit(fu13.render("No calibration!",True,C_RED),(INFO_X+6,y_i)); y_i+=15
        # Opis puzle (max 22 znaka)
        screen.blit(fu13.render(desc[:22],True,C_DIM),(INFO_X+6,y_i)); y_i+=22

        # Separator
        pygame.draw.line(screen,(60,60,60),(INFO_X+4,y_i),(INFO_X+INFO_W-4,y_i),1)
        y_i+=8

        # Mod
        mc = C_GREEN if use_et and H_calib is not None else C_DIM
        screen.blit(fu13.render(f"Mod: {mode}",True,mc),(INFO_X+6,y_i)); y_i+=18

        # Sekvenca — prelom posle 3 polja
        seq_fields = list(gaze_seq)[-6:] or ['...']
        screen.blit(fu13.render("Seq:",True,C_DIM),(INFO_X+6,y_i)); y_i+=16
        line = ""
        for sq_nm in seq_fields:
            if len(line)+len(sq_nm)+1 > 16:
                screen.blit(fu13.render(line.strip(),True,C_ORANGE),(INFO_X+10,y_i))
                y_i+=15; line=""
            line += sq_nm+" "
        if line.strip():
            screen.blit(fu13.render(line.strip(),True,C_ORANGE),(INFO_X+10,y_i)); y_i+=15

        # Top-1
        y_i = max(y_i, INFO_Y+TOP_H-38)
        screen.blit(fb13.render(f"Top-1: {top1_nm}",True,C_GREEN),(INFO_X+6,y_i)); y_i+=18

        # Kontrole
        screen.blit(fu13.render("K=calibrate ESC=exit",True,C_DIM),(INFO_X+6,y_i)); y_i+=16
        screen.blit(fu13.render("SPACE=next puzzle",True,C_DIM),(INFO_X+6,y_i))

        # Overlay posle poteza
        if show_overlay and board_before is not None:
            draw_result_overlay(screen,board_before,overlay_probs,
                                move_san_str,
                                played_sq_ov,predicted_sq_ov,fp,fl,fu,fb,
                                flipped=overlay_flipped)

        pygame.display.flip()
        clock.tick(30)


if __name__=='__main__':
    run()