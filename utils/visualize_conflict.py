import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math
import importlib
import conflicts

current_idx = [0]

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.3)

AIRCRAFT_SPEED_KTS = 280  
TIME_MINUTES = 1/3          
ARROW_LENGTH_NM = (AIRCRAFT_SPEED_KTS / 60) * TIME_MINUTES  

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

def draw_conflict(idx):
    ax.clear()
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect('equal')
    ax.set_title(f"Conflict {idx + 1}")
    ax.grid(True)
    ax.set_xlabel("X (NM)")
    ax.set_ylabel("Y (NM)")

    conflict = conflicts.conflicts[idx]
    for i, ac in enumerate(conflict["aircraft"]):
        x, y = ac["x"], ac["y"]
        heading_deg = ac["heading"]
        heading_rad = math.radians(90 - heading_deg) 

        dx = ARROW_LENGTH_NM * math.cos(heading_rad)
        dy = ARROW_LENGTH_NM * math.sin(heading_rad)

        ax.plot(x, y, 'o', color=colors[i % len(colors)], label=f"AC{i+1} FL{ac['level']}")
        ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.7,
                 fc=colors[i % len(colors)], ec=colors[i % len(colors)])

    ax.legend()
    fig.canvas.draw_idle()

def next_conflict(event):
    current_idx[0] = (current_idx[0] + 1) % len(conflicts.conflicts)
    draw_conflict(current_idx[0])

def prev_conflict(event):
    current_idx[0] = (current_idx[0] - 1) % len(conflicts.conflicts)
    draw_conflict(current_idx[0])

def refresh_conflicts(event):
    importlib.reload(conflicts)
    draw_conflict(current_idx[0])

# Buttons
axprev = plt.axes([0.1, 0.05, 0.15, 0.075])
axnext = plt.axes([0.3, 0.05, 0.15, 0.075])
axrefresh = plt.axes([0.55, 0.05, 0.25, 0.075])
btn_next = Button(axnext, 'Next')
btn_prev = Button(axprev, 'Previous')
btn_refresh = Button(axrefresh, 'Reload Conflicts')

btn_next.on_clicked(next_conflict)
btn_prev.on_clicked(prev_conflict)
btn_refresh.on_clicked(refresh_conflicts)

draw_conflict(current_idx[0])
plt.show()