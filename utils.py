import time
from IPython.display import clear_output

def render_board(state_im, delay=0.3):
    clear_output(wait=True)
    board = (state_im.squeeze() * 8).astype(int)
    symbols = {
        -1: '⬜', -2: '💣', 0: '◻️', 1: '1️⃣', 2: '2️⃣', 3: '3️⃣',
        4: '4️⃣', 5: '5️⃣', 6: '6️⃣', 7: '7️⃣', 8: '8️⃣'
    }
    print("🧠 Minesweeper Agent")
    for row in board:
        print(" ".join(symbols.get(v, '?') for v in row))
    time.sleep(delay)
