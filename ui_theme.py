"""
ui_theme.py — shared visual theme for the pygame apps.

Single source of truth for the color palette, cached fonts and small
drawing helpers (cards, pills, progress bars, key hints, buttons) used
by realtime_demo.py and collectdata.py.
"""

import pygame

# Palette — dark slate UI + classic wooden board
BG          = (15, 17, 22)
HEADER_BG   = (21, 24, 31)
CARD        = (26, 29, 37)
CARD_BORDER = (45, 50, 62)
CARD_INNER  = (33, 37, 47)

TEXT  = (230, 232, 236)
DIM   = (139, 145, 158)
FAINT = (90, 96, 108)

BOARD_LIGHT = (237, 214, 176)
BOARD_DARK  = (181, 136, 99)
BOARD_FRAME = (58, 50, 41)

GREEN  = (99, 202, 118)
RED    = (229, 96, 96)
AMBER  = (240, 190, 80)
BLUE   = (73, 129, 240)
ORANGE = (255, 145, 60)

GREEN_DARK = (30, 68, 42)
RED_DARK   = (74, 34, 34)
BLUE_DARK  = (33, 48, 82)

BTN       = (42, 47, 59)
BTN_HOVER = (56, 63, 79)

HEADER_H = 56

# Cached fonts
_fonts = {}

def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont('segoeui', size, bold=bold)
    return _fonts[key]

def piece_font(size):
    key = ('piece', size)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont('segoeuisymbol', size)
    return _fonts[key]

# Drawing helpers

def text(surf, s, pos, size=13, color=TEXT, bold=False, anchor='topleft'):
    t = font(size, bold).render(s, True, color)
    r = t.get_rect(**{anchor: pos})
    surf.blit(t, r)
    return r

def text_shadow(surf, s, pos, size=13, color=TEXT, bold=False, anchor='topleft'):
    t = font(size, bold).render(s, True, (15, 15, 18))
    r = t.get_rect(**{anchor: pos})
    surf.blit(t, r.move(1, 1))
    return text(surf, s, pos, size, color, bold, anchor)

def card(surf, rect, radius=10, bg=CARD, border=CARD_BORDER, title=None):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surf, bg, rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, 1, border_radius=radius)
    if title:
        text(surf, title.upper(), (rect.x + 12, rect.y + 9), 11, DIM, bold=True)
    return rect

def pill(surf, s, pos, bg, fg, size=11, anchor='topleft'):
    t = font(size, True).render(s, True, fg)
    r = pygame.Rect(0, 0, t.get_width() + 16, t.get_height() + 6)
    setattr(r, anchor, pos)
    pygame.draw.rect(surf, bg, r, border_radius=r.height // 2)
    surf.blit(t, t.get_rect(center=r.center))
    return r

def progress(surf, rect, frac, fg=GREEN, bg=(50, 55, 68)):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surf, bg, rect, border_radius=rect.height // 2)
    w = int(rect.width * max(0.0, min(1.0, frac)))
    if w >= rect.height:
        pygame.draw.rect(surf, fg, (rect.x, rect.y, w, rect.height),
                         border_radius=rect.height // 2)

def keycap(surf, key, pos, size=11):
    t = font(size, True).render(key, True, TEXT)
    r = pygame.Rect(0, 0, max(t.get_width() + 10, t.get_height() + 6),
                    t.get_height() + 6)
    r.topleft = pos
    pygame.draw.rect(surf, (48, 53, 66), r, border_radius=4)
    pygame.draw.rect(surf, (72, 79, 96), r, 1, border_radius=4)
    surf.blit(t, t.get_rect(center=r.center))
    return r

def hints(surf, pos, pairs, size=11, gap=16):
    """Row of `[KEY] label` hints. Returns x after the last hint."""
    x, y = pos
    for key, label in pairs:
        r = keycap(surf, key, (x, y), size)
        t = font(size).render(label, True, DIM)
        surf.blit(t, (r.right + 6, r.y + (r.height - t.get_height()) // 2))
        x = r.right + 6 + t.get_width() + gap
    return x

def chips(surf, items, pos, max_w, size=11, fg=ORANGE, bg=(50, 44, 36)):
    """Row(s) of small rounded chips, wrapping at max_w. Returns bottom y."""
    x0, y = pos
    x = x0
    row_h = 0
    for it in items:
        t = font(size, True).render(it, True, fg)
        w, h = t.get_width() + 12, t.get_height() + 4
        if x + w > x0 + max_w and x > x0:
            x = x0
            y += row_h + 4
        r = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, bg, r, border_radius=h // 2)
        surf.blit(t, t.get_rect(center=r.center))
        x += w + 5
        row_h = h
    return y + row_h

def alpha_rect(surf, rect, rgba, radius=0):
    rect = pygame.Rect(rect)
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, rgba, s.get_rect(), border_radius=radius)
    surf.blit(s, rect.topleft)

def glow_circle(surf, center, radius, color, width=2):
    """Ring with a soft outer glow."""
    pad = 8
    s = pygame.Surface((radius * 2 + pad * 2,) * 2, pygame.SRCALPHA)
    c = (radius + pad, radius + pad)
    pygame.draw.circle(s, (*color, 30), c, radius + 5)
    pygame.draw.circle(s, (*color, 70), c, radius + 2)
    pygame.draw.circle(s, (*color, 255), c, radius, width)
    surf.blit(s, (center[0] - radius - pad, center[1] - radius - pad))

def header(surf, window_w, title, subtitle=None, right_pill=None):
    pygame.draw.rect(surf, HEADER_BG, (0, 0, window_w, HEADER_H))
    pygame.draw.line(surf, CARD_BORDER, (0, HEADER_H), (window_w, HEADER_H))
    text(surf, title, (24, 8), 19, TEXT, bold=True)
    if subtitle:
        text(surf, subtitle, (24, 35), 11, DIM)
    if right_pill:
        label, bg, fg = right_pill
        pill(surf, label, (window_w - 24, HEADER_H // 2), bg, fg,
             anchor='midright')

class Button:
    def __init__(self, x, y, w, h, label, accent=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.accent = accent
        self.hover = False

    def update(self, mx, my):
        self.hover = self.rect.collidepoint(mx, my)

    def hit(self, mx, my):
        return self.rect.collidepoint(mx, my)

    def draw(self, surf):
        base = self.accent or BTN
        col = tuple(min(c + 18, 255) for c in base) if self.hover else base
        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        edge = tuple(min(c + 26, 255) for c in base)
        pygame.draw.rect(surf, edge, self.rect, 1, border_radius=8)
        t = font(14, True).render(self.label, True, TEXT)
        surf.blit(t, t.get_rect(center=self.rect.center))
