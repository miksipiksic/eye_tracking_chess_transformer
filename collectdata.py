"""
collectdata.py — training-data collection

Modes:
  GAME   — play against the bot/Stockfish, choose your color
  PUZZLE — solve lichess puzzles

  pip install pygame python-chess
  python collectdata.py

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

Use the keys so that you don't move the mouse all the time.
The mouse should follow eye movements.
"""

import pygame
import chess
import chess.engine
import random
import csv
import os
from collections import deque

import ui_theme as ui

MARGIN        = 24
BOARD_X       = MARGIN
BOARD_Y       = ui.HEADER_H + 20
BOARD_SIZE    = 560
SQUARE_SIZE   = BOARD_SIZE // 8
PANEL_X       = BOARD_X + BOARD_SIZE + 24
PANEL_W       = 220
BUTTON_W      = PANEL_W
BUTTON_H      = 40
WINDOW_W      = PANEL_X + PANEL_W + MARGIN
WINDOW_H      = BOARD_Y + BOARD_SIZE + 44
OUTPUT_CSV    = 'data/gaze_dataset.csv'
GAZE_WINDOW   = 20
BOT_DELAY_MS  = 700
STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', '')

# download lichess database .csv file
# https://database.lichess.org/#puzzles
FALLBACK_PUZZLES = [
    {'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
     'solution': 'f3g5', 'desc': 'Attack on f7'},
    {'fen': '4k3/8/4K3/7R/8/8/8/8 w - - 0 1',
     'solution': 'h5h8', 'desc': 'Mate in 1'},
    {'fen': '6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1',
     'solution': 'd1d8', 'desc': 'Back-rank mate'},
]
PUZZLES = FALLBACK_PUZZLES
PUZZLE_CSV = 'data/lichess_puzzles.csv'

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
    x = px - BOARD_X; y = py - BOARD_Y
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE): return None
    c = x // SQUARE_SIZE; r = y // SQUARE_SIZE
    return chess.square(7-c, r) if flipped else chess.square(c, 7-r)

def sq_center(sq, flipped):
    c, r = sq_col_row(sq, flipped)
    return (BOARD_X + c*SQUARE_SIZE + SQUARE_SIZE//2,
            BOARD_Y + r*SQUARE_SIZE + SQUARE_SIZE//2)


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
    Daily Puzzle from Lichess — requires an internet connection
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
            'desc':     'Daily puzzle',
            'rating':   puzzle.get('rating', 0),
            'themes':   ' '.join(puzzle.get('themes', [])),
        }
    except Exception as e:
        print(f"[PUZZLE] Can't download the daily Lichess puzzle: {e}")
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

def draw_piece(scr, fnt, sym, center, is_white):
    sh = fnt.render(sym, True, (20, 18, 16))
    scr.blit(sh, sh.get_rect(center=(center[0]+2, center[1]+2)))
    clr = (250, 250, 250) if is_white else (28, 28, 30)
    t = fnt.render(sym, True, clr)
    scr.blit(t, t.get_rect(center=center))


def draw_board(scr, board, fp, sel, last, hov, trail, flipped):
    pygame.draw.rect(scr, ui.BOARD_FRAME,
                     (BOARD_X-6, BOARD_Y-6, BOARD_SIZE+12, BOARD_SIZE+12),
                     border_radius=8)
    for sq in chess.SQUARES:
        c, r = sq_col_row(sq, flipped)
        rect = pygame.Rect(BOARD_X+c*SQUARE_SIZE,
                           BOARD_Y+r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(scr, ui.BOARD_LIGHT if (c+r)%2==0 else ui.BOARD_DARK, rect)
        if last and sq in (last.from_square, last.to_square):
            ui.alpha_rect(scr, rect, (205, 210, 55, 110))
        if sq == sel:
            ui.alpha_rect(scr, rect, (20, 200, 20, 110))
        if sq == hov and sq != sel:
            ui.alpha_rect(scr, rect, (100, 180, 255, 65))

    n = len(trail)
    for i, name in enumerate(trail):
        try: tsq = chess.parse_square(name)
        except: continue
        tc, tr2 = sq_col_row(tsq, flipped)
        ui.alpha_rect(scr, (BOARD_X+tc*SQUARE_SIZE, BOARD_Y+tr2*SQUARE_SIZE,
                            SQUARE_SIZE, SQUARE_SIZE),
                      (255, 70, 70, int(30+140*(i/max(n,1)))))

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p: continue
        sym = PIECES[p.piece_type][p.color]
        draw_piece(scr, fp, sym, sq_center(sq, flipped), p.color==chess.WHITE)

    # coordinates inside the edge squares
    for i in range(8):
        fn = chess.FILE_NAMES[i] if not flipped else chess.FILE_NAMES[7-i]
        clr = ui.BOARD_DARK if (i+7) % 2 == 0 else ui.BOARD_LIGHT
        ui.text(scr, fn, (BOARD_X+i*SQUARE_SIZE+SQUARE_SIZE-4,
                          BOARD_Y+BOARD_SIZE-2), 11, clr,
                bold=True, anchor='bottomright')
        rn = str(8-i) if not flipped else str(i+1)
        clr = ui.BOARD_DARK if i % 2 == 0 else ui.BOARD_LIGHT
        ui.text(scr, rn, (BOARD_X+3, BOARD_Y+i*SQUARE_SIZE+1), 11, clr, bold=True)


def draw_panel(scr, btns, title, items):
    """Side panel: buttons on top, then an info card.

    items: list of ('text', s, color, bold) | ('chips', [names]) |
           ('sep',) | ('pill', s, bg, fg)
    """
    for b in btns:
        b.draw(scr)
    top = (btns[-1].rect.bottom + 12) if btns else BOARD_Y
    rect = ui.card(scr, (PANEL_X, top, PANEL_W, BOARD_Y+BOARD_SIZE-top),
                   title=title)
    x = rect.x + 12
    y = rect.y + 32
    for it in items:
        kind = it[0]
        if kind == 'sep':
            pygame.draw.line(scr, ui.CARD_BORDER, (x, y+3), (rect.right-12, y+3))
            y += 12
        elif kind == 'chips':
            if it[1]:
                y = ui.chips(scr, it[1], (x, y), PANEL_W-24) + 6
            else:
                ui.text(scr, "—", (x, y), 12, ui.FAINT); y += 20
        elif kind == 'pill':
            ui.pill(scr, it[1], (x, y), it[2], it[3])
            y += 28
        else:
            _, s, color, bold = it
            ui.text(scr, s, (x, y), 12, color, bold=bold)
            y += 20


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

def run_game(scr, fp, pc, total):
    flipped = (pc == chess.BLACK)
    board = chess.Board(); sel=None; last=None; hov=None
    status = f"You are playing {'White' if pc==chess.WHITE else 'Black'}"
    gaze = deque(maxlen=GAZE_WINDOW*3); logged = None; new = 0

    bt = (board.turn != pc); bra = pygame.time.get_ticks()+BOT_DELAY_MS if bt else 0

    bx = PANEL_X
    b_new  = ui.Button(bx,BOARD_Y,   BUTTON_W,BUTTON_H,'New game',(38,92,56))
    b_undo = ui.Button(bx,BOARD_Y+48,BUTTON_W,BUTTON_H,'Undo (Z)')
    b_menu = ui.Button(bx,BOARD_Y+96,BUTTON_W,BUTTON_H,'Menu')
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
                        status=f'{san}  |  Total: {total}'
                        if board.is_game_over():
                            status=f'End: {board.result()} — new game?'
                        else:
                            bt=True; bra=now+BOT_DELAY_MS
                    else:
                        p=board.piece_at(sq)
                        sel=sq if (p and p.color==pc) else None
                        if not sel: status='Illegal move.'

        scr.fill(ui.BG)
        ui.header(scr,WINDOW_W,"Chess Gaze Collector","You vs Bot",
                  right_pill=("GAME MODE",ui.BLUE_DARK,(180,205,255)))
        draw_board(scr,board,fp,sel,last,hov,gaze,flipped)
        cl='White' if board.turn==chess.WHITE else 'Black'
        ag='YOU' if board.turn==pc else 'BOT'
        items=[('text',f'You play {"White" if pc==chess.WHITE else "Black"}',ui.TEXT,True),
               ('text',f'To move: {cl} ({ag})',ui.AMBER,False),
               ('sep',),
               ('text','GAZE (MOUSE)',ui.FAINT,True),
               ('chips',list(gaze)[-6:]),
               ('sep',),
               ('text',status,ui.GREEN,False)]
        draw_panel(scr,btns,"Game",items)
        ui.hints(scr,(BOARD_X,BOARD_Y+BOARD_SIZE+16),
                 [("N","new"),("Z","undo"),("F","flip"),("ESC","menu")])
        pygame.display.flip(); clock.tick(60)

# Puzzle mode

def run_puzzle(scr, fp, total):
    idx = 0; new = 0

    def load(i):
        p = PUZZLES[i % len(PUZZLES)]
        b = chess.Board(p['fen'])
        return b, (b.turn==chess.BLACK), p

    board, flipped, puz = load(idx)
    pc=board.turn; sel=None; last=None; hov=None
    status=f'Puzzle {idx+1}: {puz["desc"]}'
    gaze=deque(maxlen=GAZE_WINDOW*3); logged=None

    bx=PANEL_X
    b_next=ui.Button(bx,BOARD_Y,   BUTTON_W,BUTTON_H,'Next puzzle',(38,92,56))
    b_hint=ui.Button(bx,BOARD_Y+48,BUTTON_W,BUTTON_H,'Hint')
    b_menu=ui.Button(bx,BOARD_Y+96,BUTTON_W,BUTTON_H,'Menu')
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

        scr.fill(ui.BG)
        ui.header(scr,WINDOW_W,"Chess Gaze Collector","Lichess puzzles",
                  right_pill=("PUZZLE MODE",ui.GREEN_DARK,(190,255,205)))
        draw_board(scr,board,fp,sel,last,hov,gaze,flipped)
        cl='White' if board.turn==chess.WHITE else 'Black'
        items=[('text',f'Puzzle {idx+1}/{len(PUZZLES)}',ui.TEXT,True),
               ('text',puz["desc"][:26],ui.DIM,False),
               ('text',f'To move: {cl}',ui.AMBER,False),
               ('sep',),
               ('text','GAZE (MOUSE)',ui.FAINT,True),
               ('chips',list(gaze)[-6:]),
               ('sep',),
               ('text',status,ui.GREEN,False)]
        draw_panel(scr,btns,"Puzzle",items)
        ui.hints(scr,(BOARD_X,BOARD_Y+BOARD_SIZE+16),
                 [("N","next"),("H","hint"),("ESC","menu")])
        pygame.display.flip(); clock.tick(60)

# Menu

def run_menu(scr, total):
    cx=WINDOW_W//2

    b_w=ui.Button(cx-110,290,220,44,'Play White',(38,92,56))
    b_b=ui.Button(cx-110,342,220,44,'Play Black',(38,92,56))
    b_p=ui.Button(cx-110,428,220,44,'Solve puzzles',(35,55,105))
    b_q=ui.Button(cx-110,510,220,44,'Quit',(105,44,44))
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

        scr.fill(ui.BG)
        ui.text(scr,'Chess Gaze Collector',(cx,150),30,ui.TEXT,bold=True,
                anchor='center')
        ui.text(scr,'Collect gaze-sequence training data — mouse acts as gaze proxy',
                (cx,192),13,ui.DIM,anchor='center')
        ui.pill(scr,f'TOTAL MOVES COLLECTED: {total}',(cx,228),
                ui.GREEN_DARK,(190,255,205),anchor='center')
        ui.text(scr,'YOU vs BOT',(cx,272),11,ui.FAINT,bold=True,anchor='center')
        ui.text(scr,'PUZZLES',(cx,412),11,ui.FAINT,bold=True,anchor='center')
        for b in btns: b.draw(scr)
        ui.hints(scr,(MARGIN,WINDOW_H-32),
                 [("1/W","white"),("2/B","black"),("3/P","puzzles"),
                  ("Q/ESC","quit")])
        pygame.display.flip(); clock.tick(60)


def count_saved():
    if not os.path.exists(OUTPUT_CSV): return 0
    with open(OUTPUT_CSV,'r',encoding='utf-8') as f:
        return max(0, sum(1 for _ in f)-1)

if __name__ == '__main__':
    pygame.init()
    scr = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption('Chess Gaze Collector')
    fp = ui.piece_font(SQUARE_SIZE-10)
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
        mod, pc = run_menu(scr, total)
        if mod is None: break
        if mod == 'game':   new, sig = run_game(scr, fp, pc, total)
        elif mod == 'puzzle': new, sig = run_puzzle(scr, fp, total)
        total += new
        if sig == 'quit': break

    pygame.quit()
    print(f'Total moves: {total} → {OUTPUT_CSV}')
