"""
collect_data.py — training

Modes:
  GAME   — against the bot/stockfish - choose the color
  PUZZLE — solve lichess puzzles

  pip install pygame python-chess
  python src/collect_data.py

Keys — GAME mode:
  N       — new game
  Z       — undo move (your move + bot move)
  ESC     — back to menu
  F       — flip board 

Keys — PUZZLE mode:
  ENTER / N  — next puzzle
  H          — hint
  ESC        — menu

Keys — MENU:
  1 / W   — play white
  2 / B   — play black
  3 / P   — puzzles
  ESC / Q — exit

Use keys so that you don't move the mouse all the time. 
The mouse should follow eye movements.
"""

import pygame
import chess
import chess.engine
import random
import csv
import os
from collections import deque

WINDOW_W      = 860
WINDOW_H      = 680
BOARD_OFFSET  = 40
BOARD_SIZE    = 560
SQUARE_SIZE   = BOARD_SIZE // 8
PANEL_X       = BOARD_OFFSET + BOARD_SIZE + 20
BUTTON_W      = 160
BUTTON_H      = 36
OUTPUT_CSV    = 'data/gaze_dataset.csv'
GAZE_WINDOW   = 20
BOT_DELAY_MS  = 700
STOCKFISH_PATH = "STOCKFISH_PATH"

# download lichess database .csv file
# https://database.lichess.org/#puzzles
FALLBACK_PUZZLES = [
    {'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
     'solution': 'f3g5', 'desc': 'Napad na f7'},
    {'fen': '4k3/8/4K3/4R3/8/8/8/8 w - - 0 1',
     'solution': 'e5e8', 'desc': 'Mat u 1'},
    {'fen': 'r4rk1/ppp2ppp/2n5/3p4/3P4/2N5/PPP2PPP/R4RK1 w - - 0 1',
     'solution': 'f1f8', 'desc': 'Napad topom'},
]
PUZZLES = FALLBACK_PUZZLES   
PUZZLE_CSV = 'data/lichess_puzzles.csv'  

C_LIGHT = (240,217,181); C_DARK = (181,136,99); C_BG = (30,30,30)
C_PANEL = (45,45,45);    C_TEXT = (220,220,220); C_DIM = (140,140,140)
C_GREEN = (80,200,80);   C_BLUE = (80,140,220);  C_YELLOW = (220,200,60)
C_RED   = (220,80,80);   C_BTN  = (65,65,65);    C_BTN_H = (90,90,90)
C_ACT   = (50,130,50)

PIECES = {
    chess.PAWN:   {chess.WHITE:'♙', chess.BLACK:'♟'},
    chess.KNIGHT: {chess.WHITE:'♘', chess.BLACK:'♞'},
    chess.BISHOP: {chess.WHITE:'♗', chess.BLACK:'♝'},
    chess.ROOK:   {chess.WHITE:'♖', chess.BLACK:'♜'},
    chess.QUEEN:  {chess.WHITE:'♕', chess.BLACK:'♛'},
    chess.KING:   {chess.WHITE:'♔', chess.BLACK:'♚'},
}


# Coords

def sq_col_row(sq, flipped):
    f = chess.square_file(sq); r = chess.square_rank(sq)
    return (7-f, r) if flipped else (f, 7-r)

def px_to_sq(px, py, flipped):
    x = px - BOARD_OFFSET; y = py - BOARD_OFFSET
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE): return None
    c = x // SQUARE_SIZE; r = y // SQUARE_SIZE
    return chess.square(7-c, r) if flipped else chess.square(c, 7-r)

def sq_center(sq, flipped):
    c, r = sq_col_row(sq, flipped)
    return (BOARD_OFFSET + c*SQUARE_SIZE + SQUARE_SIZE//2,
            BOARD_OFFSET + r*SQUARE_SIZE + SQUARE_SIZE//2)



class Btn:
    def __init__(self, x, y, w, h, lbl, col=None):
        self.r = pygame.Rect(x, y, w, h)
        self.lbl = lbl; self.col = col or C_BTN; self.hov = False
    def update(self, mx, my): self.hov = self.r.collidepoint(mx, my)
    def hit(self, mx, my): return self.r.collidepoint(mx, my)
    def draw(self, scr, fnt):
        pygame.draw.rect(scr, C_BTN_H if self.hov else self.col,
                         self.r, border_radius=6)
        pygame.draw.rect(scr, (80,80,80), self.r, 1, border_radius=6)
        t = fnt.render(self.lbl, True, C_TEXT)
        scr.blit(t, t.get_rect(center=self.r.center))



def load_puzzles(csv_path=PUZZLE_CSV, n=200, min_rating=1000, max_rating=1800):
    """
        list of dict {'fen', 'solution', 'desc', 'rating', 'themes'}
    """
    if not os.path.exists(csv_path):
        print(f"[PUZZLE] Database not found: {csv_path}")
        print( "[PUZZLE] Using fallback puzzles.")
        print( "[PUZZLE] Download lichess_puzzles.csv.zst from")
        print( "[PUZZLE]   https://database.lichess.org/#puzzles")
        print( "[PUZZLE] and export to data/lichess_puzzles.csv")
        return None

    import csv as _csv

    def parse_row(row):
        try:
            rating = int(row.get('Rating', 0))
            if not (min_rating <= rating <= max_rating):
                return None
            moves = row['Moves'].strip().split()
            if len(moves) < 2:
                return None
            board_tmp = chess.Board(row['FEN'])
            setup_move = chess.Move.from_uci(moves[0])
            if setup_move not in board_tmp.legal_moves:
                return None
            board_tmp.push(setup_move)
            themes = row.get('Themes', '').strip()
            desc   = themes.split()[0] if themes else f"Rating {rating}"
            return {
                'fen':       board_tmp.fen(),
                'solution':  moves[1],
                'all_moves': moves[1:],
                'desc':      desc,
                'rating':    rating,
                'themes':    themes,
            }
        except Exception:
            return None

    try:
        # Pick puzzles based on rating range
        print(f"[PUZZLE] Scanning database...")
        valid_indices = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    rating = int(row.get('Rating', 0))
                    if min_rating <= rating <= max_rating:
                        valid_indices.append(i)
                except Exception:
                    continue

        if not valid_indices:
            print("[PUZZLE] Puzzles in range not found.")
            return None

        print(f"[PUZZLE] Found {len(valid_indices):,} puzzles in range "
              f"{min_rating}–{max_rating}.")

        # Pick n random puzzles
        sample_size = min(n * 3, len(valid_indices))
        sampled = set(random.sample(valid_indices, sample_size))

        # take only n rows
        puzzles = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for i, row in enumerate(reader):
                if i not in sampled:
                    continue
                p = parse_row(row)
                if p:
                    puzzles.append(p)
                if len(puzzles) >= n:
                    break

        random.shuffle(puzzles)
        print(f"[PUZZLE] Loaded {len(puzzles)} puzzles from {csv_path}")
        return puzzles if puzzles else None

    except Exception as e:
        print(f"[PUZZLE] ERROR database not found: {e}")
        return None


def fetch_daily_puzzle():
    """
    Daily Puzzle from Lichess - required internet conenction
    """
    try:
        import urllib.request, json as _json
        url = 'https://lichess.org/api/puzzle/daily'
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = _json.loads(r.read().decode())

        puzzle = data.get('puzzle', {})
        game   = data.get('game', {})
        moves  = puzzle.get('solution', [])

        if not moves:
            return None

        # FEN + initialPly 
        pgn_moves = game.get('pgn', '').split()
        init_ply  = puzzle.get('initialPly', 0)
        board_tmp = chess.Board()
        for i, mv in enumerate(pgn_moves):
            if i >= init_ply:
                break
            try:
                board_tmp.push_san(mv)
            except Exception:
                break

        return {
            'fen':      board_tmp.fen(),
            'solution': moves[0],
            'all_moves': moves,
            'desc':     'Dnevna puzla',
            'rating':   puzzle.get('rating', 0),
            'themes':   ' '.join(puzzle.get('themes', [])),
        }
    except Exception as e:
        print(f"[PUZZLE] Can't download Daily Lichess Puzzle: {e}")
        return None


# Bot

def bot_move(board):
    if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
        try:
            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as e:
                return e.play(board, chess.engine.Limit(time=0.5)).move
        except: pass
    mvs = list(board.legal_moves)
    return random.choice(mvs) if mvs else None


# Drawing

def draw_board(scr, board, fp, fl, sel, last, hov, trail, flipped):
    for sq in chess.SQUARES:
        c, r = sq_col_row(sq, flipped)
        rect = pygame.Rect(BOARD_OFFSET+c*SQUARE_SIZE,
                           BOARD_OFFSET+r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(scr, C_LIGHT if (c+r)%2==0 else C_DARK, rect)
        if last and sq in (last.from_square, last.to_square):
            s=pygame.Surface((SQUARE_SIZE,SQUARE_SIZE),pygame.SRCALPHA)
            s.fill((205,210,55,110)); scr.blit(s, rect.topleft)
        if sq == sel:
            s=pygame.Surface((SQUARE_SIZE,SQUARE_SIZE),pygame.SRCALPHA)
            s.fill((20,200,20,150)); scr.blit(s, rect.topleft)
        if sq == hov and sq != sel:
            s=pygame.Surface((SQUARE_SIZE,SQUARE_SIZE),pygame.SRCALPHA)
            s.fill((100,180,255,65)); scr.blit(s, rect.topleft)

    n = len(trail)
    for i, name in enumerate(trail):
        try: tsq = chess.parse_square(name)
        except: continue
        tc, tr2 = sq_col_row(tsq, flipped)
        s=pygame.Surface((SQUARE_SIZE,SQUARE_SIZE),pygame.SRCALPHA)
        s.fill((255,70,70, int(30+140*(i/max(n,1)))))
        scr.blit(s,(BOARD_OFFSET+tc*SQUARE_SIZE, BOARD_OFFSET+tr2*SQUARE_SIZE))

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p: continue
        sym = PIECES[p.piece_type][p.color]
        cx, cy = sq_center(sq, flipped)
        sh = fp.render(sym, True, (0,0,0))
        scr.blit(sh, sh.get_rect(center=(cx+2,cy+2)))
        scr.blit(fp.render(sym, True,
            (255,255,255) if p.color==chess.WHITE else (25,25,25)),
            fp.render(sym,True,(0,0,0)).get_rect(center=(cx,cy)))

    for i in range(8):
        fi = chess.FILE_NAMES[i] if not flipped else chess.FILE_NAMES[7-i]
        ri = str(8-i) if not flipped else str(i+1)
        scr.blit(fl.render(fi,True,C_TEXT),(BOARD_OFFSET+i*SQUARE_SIZE+3,
                                             BOARD_OFFSET+BOARD_SIZE+4))
        scr.blit(fl.render(ri,True,C_TEXT),(BOARD_OFFSET-18,
                  BOARD_OFFSET+i*SQUARE_SIZE+SQUARE_SIZE//2-8))
    pygame.draw.rect(scr,(70,70,70),
        (BOARD_OFFSET,BOARD_OFFSET,BOARD_SIZE,BOARD_SIZE),2)


def draw_panel(scr, fnt, btns, lines):
    pygame.draw.rect(scr, C_PANEL, (PANEL_X-10,0,WINDOW_W-PANEL_X+10,WINDOW_H))
    for b in btns: b.draw(scr, fnt)
    y = btns[-1].r.bottom + 20 if btns else BOARD_OFFSET
    for txt, col in lines:
        if txt == '---':
            pygame.draw.line(scr,(70,70,70),(PANEL_X,y+4),(PANEL_X+BUTTON_W+20,y+4))
            y += 14; continue
        scr.blit(fnt.render(txt,True,col),(PANEL_X,y)); y += 22


# Saving

def save_rec(rec):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    exists = os.path.exists(OUTPUT_CSV)
    fields = ['gaze_sequence','move_uci','move_san','move_number',
              'color','fen','target_square','mode']
    with open(OUTPUT_CSV,'a',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow(rec)


# Game mode

def run_game(scr, fonts, pc, total):
    fp, fl, fu = fonts
    flipped = (pc == chess.BLACK)
    board = chess.Board(); sel=None; last=None; hov=None
    status = f"You are playing {'White' if pc==chess.WHITE else 'Black'}"
    gaze = deque(maxlen=GAZE_WINDOW*3); logged = None; new = 0

    bt = (board.turn != pc); bra = pygame.time.get_ticks()+BOT_DELAY_MS if bt else 0

    bx = PANEL_X
    b_new  = Btn(bx,60, BUTTON_W,BUTTON_H,'New game',C_ACT)
    b_undo = Btn(bx,106,BUTTON_W,BUTTON_H,'Undo (Z)')
    b_menu = Btn(bx,152,BUTTON_W,BUTTON_H,'Menu')
    btns = [b_new, b_undo, b_menu]
    clock = pygame.time.Clock()

    while True:
        mx,my = pygame.mouse.get_pos(); now = pygame.time.get_ticks()
        hov = px_to_sq(mx,my,flipped)
        if hov is not None and hov != logged:
            gaze.append(chess.square_name(hov)); logged = hov
        for b in btns: b.update(mx,my)

        if bt and not board.is_game_over() and board.turn!=pc and now>=bra:
            mv = bot_move(board)
            if mv: board.push(mv); last = mv
            bt = False; status = 'Your turn.'
            pygame.event.clear(pygame.MOUSEBUTTONDOWN)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return new,'quit'
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return new,'menu'
                elif ev.key in (pygame.K_z, pygame.K_u):
                    if len(board.move_stack)>=2:
                        board.pop(); board.pop(); sel=None
                        last = board.peek() if board.move_stack else None
                        status='Undo.'
                elif ev.key == pygame.K_n:
                    board=chess.Board(); sel=None; last=None; gaze.clear()
                    logged=None; bt=(board.turn!=pc)
                    bra=now+BOT_DELAY_MS; status='New game!'
                elif ev.key == pygame.K_f:
                    flipped = not flipped; status='Board flipped.'
                elif ev.key == pygame.K_m:
                    return new,'menu'
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button==1:
                if b_new.hit(mx,my):
                    board=chess.Board(); sel=None; last=None; gaze.clear()
                    logged=None; bt=(board.turn!=pc)
                    bra=now+BOT_DELAY_MS; status='New game!'; continue
                if b_undo.hit(mx,my):
                    if len(board.move_stack)>=2:
                        board.pop(); board.pop(); sel=None
                        last=board.peek() if board.move_stack else None
                        status='Undo.'; continue
                if b_menu.hit(mx,my): return new,'menu'
                if bt or board.is_game_over() or board.turn!=pc: continue
                sq = px_to_sq(mx,my,flipped)
                if sq is None: sel=None; continue
                if sel is None:
                    p=board.piece_at(sq)
                    if p and p.color==pc: sel=sq
                else:
                    mv = chess.Move(sel,sq)
                    p = board.piece_at(sel)
                    if p and p.piece_type==chess.PAWN and chess.square_rank(sq) in (0,7):
                        mv = chess.Move(sel,sq,promotion=chess.QUEEN)
                    if mv in board.legal_moves:
                        snap=list(gaze).copy(); fen0=board.fen()
                        san=board.san(mv); uci=mv.uci()
                        clr='white' if board.turn==chess.WHITE else 'black'
                        mn=board.fullmove_number
                        board.push(mv); last=mv; sel=None
                        gaze.clear(); logged=None
                        rec={'gaze_sequence':','.join(snap),'move_uci':uci,
                             'move_san':san,'move_number':mn,'color':clr,
                             'fen':fen0,'target_square':chess.square_name(sq),
                             'mode':'game'}
                        save_rec(rec); new+=1; total+=1
                        status=f'{san}  |  Ukupno: {total}'
                        if board.is_game_over():
                            status=f'End: {board.result()} — new game?'
                        else:
                            bt=True; bra=now+BOT_DELAY_MS
                    else:
                        p=board.piece_at(sq)
                        sel=sq if (p and p.color==pc) else None
                        if not sel: status='Ilegal move.'

        scr.fill(C_BG)
        draw_board(scr,board,fp,fl,sel,last,hov,gaze,flipped)
        cl='White' if board.turn==chess.WHITE else 'Black'
        ag='YOU' if board.turn==pc else 'BOT'
        info=[(f'Mode: You vs BOT',C_DIM),(f'You: {"White" if pc==chess.WHITE else "Black"}  |  F=flip',C_TEXT),
              ('---',C_TEXT),(f'To move: {cl} ({ag})',C_YELLOW),
              ('---',C_TEXT),('Gaze (mouse):',C_DIM),
              (' → '.join(list(gaze)[-6:]) or '—', C_RED),
              ('---',C_TEXT),('N=new  Z=undo  M=menu',C_DIM),
              ('F=flip  ESC=menu',C_DIM),
              ('---',C_TEXT),(status,C_GREEN)]
        draw_panel(scr,fu,btns,info)
        pygame.display.flip(); clock.tick(60)

# Puzzle mode

def run_puzzle(scr, fonts, total):
    fp, fl, fu = fonts
    idx = 0; new = 0

    def load(i):
        p = PUZZLES[i % len(PUZZLES)]
        b = chess.Board(p['fen'])
        return b, (b.turn==chess.BLACK), p

    board, flipped, puz = load(idx)
    pc=board.turn; sel=None; last=None; hov=None
    status=f'Puzla {idx+1}: {puz["desc"]}'
    gaze=deque(maxlen=GAZE_WINDOW*3); logged=None

    bx=PANEL_X
    b_next=Btn(bx,60, BUTTON_W,BUTTON_H,'Next puzzle',C_ACT)
    b_hint=Btn(bx,106,BUTTON_W,BUTTON_H,'Hint')
    b_menu=Btn(bx,152,BUTTON_W,BUTTON_H,'Menu')
    btns=[b_next,b_hint,b_menu]
    clock=pygame.time.Clock()

    while True:
        mx,my=pygame.mouse.get_pos()
        hov=px_to_sq(mx,my,flipped)
        if hov is not None and hov!=logged:
            gaze.append(chess.square_name(hov)); logged=hov
        for b in btns: b.update(mx,my)

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: return new,'quit'
            elif ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE: return new,'menu'
                elif ev.key in (pygame.K_RETURN, pygame.K_n):
                    idx+=1; board,flipped,puz=load(idx); pc=board.turn
                    sel=None; last=None; gaze.clear(); logged=None
                    status=f'Puzzle {idx+1}: {puz["desc"]}'
                elif ev.key == pygame.K_h:
                    status=f'Hint: {puz["solution"]}'
                elif ev.key == pygame.K_m:
                    return new,'menu'
            elif ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                if b_next.hit(mx,my):
                    idx+=1; board,flipped,puz=load(idx); pc=board.turn
                    sel=None; last=None; gaze.clear(); logged=None
                    status=f'Puzzle {idx+1}: {puz["desc"]}'; continue
                if b_hint.hit(mx,my):
                    status=f'Hint: {puz["solution"]}'; continue
                if b_menu.hit(mx,my): return new,'menu'
                sq=px_to_sq(mx,my,flipped)
                if sq is None or board.is_game_over(): sel=None; continue
                if sel is None:
                    p=board.piece_at(sq)
                    if p and p.color==pc: sel=sq
                else:
                    mv=chess.Move(sel,sq)
                    p=board.piece_at(sel)
                    if p and p.piece_type==chess.PAWN and chess.square_rank(sq) in (0,7):
                        mv=chess.Move(sel,sq,promotion=chess.QUEEN)
                    if mv in board.legal_moves:
                        snap=list(gaze).copy(); fen0=board.fen()
                        san=board.san(mv); uci=mv.uci()
                        clr='white' if board.turn==chess.WHITE else 'black'
                        mn=board.fullmove_number
                        board.push(mv); last=mv; sel=None
                        gaze.clear(); logged=None
                        ok = (uci==puz['solution'])
                        status=f'{"Correct!" if ok else "Wrong"} — {san}'
                        save_rec({'gaze_sequence':','.join(snap),'move_uci':uci,
                                  'move_san':san,'move_number':mn,'color':clr,
                                  'fen':fen0,'target_square':chess.square_name(sq),
                                  'mode':'puzzle'})
                        new+=1; total+=1
                    else:
                        p=board.piece_at(sq)
                        sel=sq if (p and p.color==pc) else None

        scr.fill(C_BG)
        draw_board(scr,board,fp,fl,sel,last,hov,gaze,flipped)
        cl='White' if board.turn==chess.WHITE else 'Black'
        info=[(f'Mode: Puzzle  {idx+1}/{len(PUZZLES)}',C_DIM),
              (f'{puz["desc"]}',C_TEXT),('---',C_TEXT),
              (f'To move: {cl}',C_YELLOW),('(board flipped)',C_DIM),
              ('---',C_TEXT),('Gaze (mouse):',C_DIM),
              (' → '.join(list(gaze)[-6:]) or '—', C_RED),
              ('---',C_TEXT),('ENTER/N=next  H=hint',C_DIM),
              ('M=menu  ESC=menu',C_DIM),
              ('---',C_TEXT),(status,C_GREEN)]
        draw_panel(scr,fu,btns,info)
        pygame.display.flip(); clock.tick(60)

# Menu

def run_menu(scr, fonts, total):
    _,_,fu=fonts
    ft=pygame.font.SysFont('arial',28,bold=True)
    fs=pygame.font.SysFont('arial',15)
    cx=WINDOW_W//2

    b_w=Btn(cx-90,210,180,44,'Play White',C_ACT)
    b_b=Btn(cx-90,266,180,44,'Play Black',C_ACT)
    b_p=Btn(cx-90,350,180,44,'Solve puzzles',C_BLUE)
    b_q=Btn(cx-90,430,180,44,'EXIT',       (130,50,50))
    btns=[b_w,b_b,b_p,b_q]
    clock=pygame.time.Clock()

    while True:
        mx,my=pygame.mouse.get_pos()
        for b in btns: b.update(mx,my)
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: return None,None
            elif ev.type==pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q): return None,None
                elif ev.key in (pygame.K_1, pygame.K_w):    return 'game',chess.WHITE
                elif ev.key in (pygame.K_2, pygame.K_b):    return 'game',chess.BLACK
                elif ev.key in (pygame.K_3, pygame.K_p):    return 'puzzle',None
            elif ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                if b_w.hit(mx,my): return 'game',chess.WHITE
                if b_b.hit(mx,my): return 'game',chess.BLACK
                if b_p.hit(mx,my): return 'puzzle',None
                if b_q.hit(mx,my): return None,None

        scr.fill(C_BG)
        scr.blit(ft.render('Chess Gaze Collector',True,C_TEXT),
                 ft.render('Chess Gaze Collector',True,C_TEXT).get_rect(center=(cx,100)))
        scr.blit(fs.render('Mouse = gaze proxy',True,C_DIM),
                 fs.render('Mouse - gaze proxy',True,C_DIM).get_rect(center=(cx,140)))
        scr.blit(fs.render(f'Total moves: {total}',True,C_GREEN),
                 fs.render(f'Total moves: {total}',True,C_GREEN).get_rect(center=(cx,168)))
        scr.blit(fu.render('YOU vs BOT─',True,C_DIM),
                 fu.render('YOU vs BOT',True,C_DIM).get_rect(center=(cx,192)))
        scr.blit(fu.render('Puzzles',True,C_DIM),
                 fu.render('Puzzles',True,C_DIM).get_rect(center=(cx,330)))
        for b in btns: b.draw(scr,fu)
        scr.blit(fs.render('1/W=White  2/B=Black  3/P=Puzzles  Q/ESC=EXIT',True,C_DIM),(20,WINDOW_H-24))
        pygame.display.flip(); clock.tick(60)



def count_saved():
    if not os.path.exists(OUTPUT_CSV): return 0
    with open(OUTPUT_CSV,'r',encoding='utf-8') as f:
        return max(0, sum(1 for _ in f)-1)

if __name__ == '__main__':
    pygame.init()
    scr = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption('Chess Gaze Collector')
    fonts = (pygame.font.SysFont('segoeuisymbol', SQUARE_SIZE-10),
             pygame.font.SysFont('arial',13),
             pygame.font.SysFont('arial',14))
    os.makedirs('data', exist_ok=True)
    total = count_saved()

    # load lichess puzzles
    loaded = load_puzzles()
    if loaded:
        PUZZLES[:] = loaded
    else:
        daily = fetch_daily_puzzle()
        if daily:
            PUZZLES[:] = [daily] + FALLBACK_PUZZLES
            print("[PUZZLE] Daily puzzle downloaded.")
        else:
            PUZZLES[:] = FALLBACK_PUZZLES
            print("[PUZZLE] Using fallback puzzles.")

    while True:
        mod, pc = run_menu(scr, fonts, total)
        if mod is None: break
        if mod == 'game':   new, sig = run_game(scr, fonts, pc, total)
        elif mod == 'puzzle': new, sig = run_puzzle(scr, fonts, total)
        total += new
        if sig == 'quit': break

    pygame.quit()
    print(f'Total moves: {total} → {OUTPUT_CSV}')