import serial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from collections import deque
import json
import os
import time
import sys
import math

# ==========================================
#        GLOBAL CONFIGURATION & FLAGS
# ==========================================

# --- FLAGS ---
motor_running = True
plot_running = True
show_labels = False
is_closing = False

# --- ZOOM / INTERACTION FLAGS ---
is_zoom_mode = False
zoom_start_pos = None 
last_mouse_x = 0
last_mouse_y = 0
is_dragging = False

# --- SYSTEM CONSTANTS ---
DISTANCE_PER_CYCLE = 3.45 
SETTINGS_FILE = "lidar_settings.json"

# --- SPEED CALCULATION VARS ---
raw_hall_count = 0      
count_offset = 0        
last_pulse_time = 0      
current_speed_kmph = 0.0  

# --- DATA STORAGE ---
scan_history = deque(maxlen=25)
sps_counter = 0
data_lock = threading.Lock()

# ==========================================
#           LOGIC HELPER FUNCTIONS
# ==========================================

def get_uint16_le(b):
    return b[0] | (b[1]<<8)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'range': 10000, 'history': 25, 'point_size': 2.0,
        'angle_from': 0, 'angle_to': 359, 'rotation': 0
    }

def save_settings():
    settings = {
        'range': range_var.get(),
        'history': history_var.get(),
        'point_size': point_size_var.get(),
        'angle_from': angle_from_var.get(),
        'angle_to': angle_to_var.get(),
        'rotation': rotation_var.get()
    }
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f)
    except:
        pass

# ==========================================
#           SERIAL READ THREAD
# ==========================================

def serial_read_thread():
    global scan_history, sps_counter, raw_hall_count, last_pulse_time, current_speed_kmph
    try:
        # === CHECK YOUR COM PORT ===
        ser = serial.Serial('COM9', 128000, timeout=0.005) 
    except Exception as e:
        print(f"Serial Error: {e}")
        return

    buffer = b''
    while not is_closing:
        if not motor_running:
            time.sleep(0.1)
            continue
            
        try:
            new_data = ser.read(8192)
        except:
            break
            
        if new_data:
            buffer += new_data
        
        # --- HALL SENSOR PARSING ---
        try:
            cnt_idx = buffer.find(b'CNT:')
            if cnt_idx != -1:
                eol_idx = buffer.find(b'\n', cnt_idx)
                if eol_idx != -1:
                    val_str = buffer[cnt_idx+4:eol_idx].decode(errors='ignore').strip()
                    if val_str.isdigit():
                        new_val = int(val_str)
                        if new_val != raw_hall_count:
                            now = time.time()
                            if last_pulse_time > 0:
                                time_diff = now - last_pulse_time
                                if time_diff > 0.05: 
                                    speed_mps = DISTANCE_PER_CYCLE / time_diff
                                    current_speed_kmph = speed_mps * 3.6
                            last_pulse_time = now
                            raw_hall_count = new_val
        except Exception:
            pass

        # --- LIDAR PACKET PARSING ---
        processed = 0
        while processed < 100:
            start = buffer.find(b'\xAA\x55')
            if start == -1: break
            if start > 0: buffer = buffer[start:]
            if len(buffer) < 10: break
            
            lsn = buffer[3]
            packet_len = 10 + lsn * 2
            if len(buffer) < packet_len: break
            
            packet = buffer[:packet_len]
            fsa = get_uint16_le(packet[4:6]) / 100.0
            lsa = get_uint16_le(packet[6:8]) / 100.0

            temp_angles = []
            temp_dists = []
            
            for i in range(lsn):
                idx = 10 + i * 2
                dist_mm = get_uint16_le(packet[idx:idx+2])
                if lsn > 1:
                    angle_deg = (fsa + (lsa-fsa) * i / (lsn-1)) % 360.0
                else:
                    angle_deg = fsa % 360.0
                
                if dist_mm > 0:
                    temp_angles.append(angle_deg)
                    temp_dists.append(dist_mm)
            
            if temp_angles:
                with data_lock:
                    scan_history.append((temp_angles, temp_dists))
                    sps_counter += len(temp_angles)
            
            buffer = buffer[packet_len:]
            processed += 1

# ==========================================
#           GUI SETUP
# ==========================================

C_BG = "#121212"       
C_PANEL = "#1E1E1E"    
C_ACCENT = "#00E5FF"   
C_WARN = "#FF5252"     
C_OK = "#00E676"       
C_SELECT = "#FFD740"   

root = tk.Tk()
root.title("YDLIDAR X4 Pro - Unified Dashboard")
root.geometry("1280x800")
root.configure(bg=C_BG)

saved_settings = load_settings()

range_var = tk.DoubleVar(value=saved_settings['range'])
history_var = tk.DoubleVar(value=saved_settings['history'])
point_size_var = tk.DoubleVar(value=saved_settings['point_size'])
angle_from_var = tk.DoubleVar(value=saved_settings['angle_from'])
angle_to_var = tk.DoubleVar(value=saved_settings['angle_to'])
rotation_var = tk.DoubleVar(value=saved_settings['rotation'])

var_speed = tk.StringVar(value="0.0")
var_dist = tk.StringVar(value="0.00")
var_cycles = tk.StringVar(value="0")
var_sps = tk.StringVar(value="0")
var_status = tk.StringVar(value="SYSTEM READY")

sidebar = tk.Frame(root, bg=C_PANEL, width=280, padx=15, pady=15)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False) 

main_area = tk.Frame(root, bg=C_BG)
main_area.pack(side="right", fill="both", expand=True)

hud_frame = tk.Frame(main_area, bg=C_BG, height=100, pady=10)
hud_frame.pack(side="top", fill="x")

graph_frame = tk.Frame(main_area, bg="black", bd=1, relief="solid")
graph_frame.pack(side="top", fill="both", expand=True, padx=20, pady=20)

def create_card(parent, title, value_var, unit, color):
    card = tk.Frame(parent, bg=C_PANEL, padx=10, pady=5)
    card.pack(side="left", fill="y", expand=True, padx=10)
    tk.Label(card, text=title, bg=C_PANEL, fg="#AAAAAA", font=("Arial", 8, "bold")).pack(anchor="w")
    val_box = tk.Frame(card, bg=C_PANEL)
    val_box.pack(pady=2)
    tk.Label(val_box, textvariable=value_var, bg=C_PANEL, fg=color, font=("Consolas", 26, "bold")).pack(side="left")
    tk.Label(val_box, text=unit, bg=C_PANEL, fg="#888888", font=("Arial", 10)).pack(side="left", padx=(5,0), anchor="s")

def create_slider(parent, label, var, min_v, max_v, cmd=None):
    f = tk.Frame(parent, bg=C_PANEL)
    f.pack(fill="x", pady=8)
    header = tk.Frame(f, bg=C_PANEL)
    header.pack(fill="x")
    tk.Label(header, text=label, bg=C_PANEL, fg="white", font=("Arial", 9)).pack(side="left")
    val_lbl = tk.Label(header, text=f"{var.get():.0f}", bg=C_PANEL, fg=C_ACCENT, font=("Arial", 9, "bold"))
    val_lbl.pack(side="right")
    
    def on_move(v):
        val_lbl.config(text=f"{float(v):.0f}")
        if cmd: cmd(float(v))
        save_settings()

    s = tk.Scale(f, variable=var, from_=min_v, to=max_v, orient="horizontal", 
             bg=C_PANEL, fg="white", highlightthickness=0, troughcolor="#333333",
             activebackground=C_ACCENT, showvalue=0, command=on_move)
    s.pack(fill="x", pady=(2,0))

create_card(hud_frame, "SPEED", var_speed, "KM/H", C_ACCENT)
create_card(hud_frame, "DISTANCE", var_dist, "M", "#E040FB")
create_card(hud_frame, "CYCLES", var_cycles, "#", "#FFEA00")
create_card(hud_frame, "DATA RATE", var_sps, "SPS", "white")

tk.Label(sidebar, text="YDLIDAR CONTROL", bg=C_PANEL, fg="white", font=("Arial", 14, "bold")).pack(pady=(0, 20), anchor="w")
tk.Label(sidebar, text="MOTOR CONTROL", bg=C_PANEL, fg="#888888", font=("Arial", 8, "bold")).pack(anchor="w")
btn_box = tk.Frame(sidebar, bg=C_PANEL)
btn_box.pack(fill="x", pady=5)

def start_m(): global motor_running, plot_running; motor_running=True; plot_running=True; var_status.set("RUNNING"); status_lbl.config(fg=C_OK)
def stop_m(): global motor_running; motor_running=False; var_status.set("STOPPED"); status_lbl.config(fg=C_WARN); save_settings()
def rst_dst(): global count_offset, current_speed_kmph; count_offset = raw_hall_count; current_speed_kmph=0.0

b_style = {"relief": "flat", "font": ("Arial", 9, "bold"), "fg": "black"}
tk.Button(btn_box, text="START", bg=C_OK, command=start_m, **b_style).pack(side="left", fill="x", expand=True, padx=2)
tk.Button(btn_box, text="STOP", bg=C_WARN, command=stop_m, **b_style).pack(side="left", fill="x", expand=True, padx=2)
tk.Button(sidebar, text="RESET DISTANCE", bg="#37474F", fg="white", relief="flat", command=rst_dst).pack(fill="x", pady=5)

tk.Frame(sidebar, height=1, bg="#333333").pack(fill="x", pady=15)

tk.Label(sidebar, text="INTERACTION", bg=C_PANEL, fg="#888888", font=("Arial", 8, "bold")).pack(anchor="w")

def toggle_zoom_mode():
    global is_zoom_mode
    is_zoom_mode = not is_zoom_mode
    if is_zoom_mode:
        zoom_btn.config(bg=C_SELECT, text="DRAG TO SELECT")
        root.config(cursor="cross")
    else:
        zoom_btn.config(bg="#37474F", text="SELECT AREA (ZOOM)")
        root.config(cursor="")

def reset_view():
    range_var.set(10000)
    angle_from_var.set(0)
    angle_to_var.set(359)
    save_settings()

zoom_btn = tk.Button(sidebar, text="SELECT AREA (ZOOM)", bg="#37474F", fg="white", relief="flat", command=toggle_zoom_mode)
zoom_btn.pack(fill="x", pady=2)
tk.Button(sidebar, text="RESET VIEW", bg="#37474F", fg="white", relief="flat", command=reset_view).pack(fill="x", pady=2)

tk.Frame(sidebar, height=1, bg="#333333").pack(fill="x", pady=15)

create_slider(sidebar, "Max Range (mm)", range_var, 500, 20000)
create_slider(sidebar, "Point Size", point_size_var, 0.5, 8.0)
create_slider(sidebar, "History (Trails)", history_var, 10, 100, lambda v: scan_history.clear())

tk.Frame(sidebar, height=1, bg="#333333").pack(fill="x", pady=15)

create_slider(sidebar, "Rotation (°)", rotation_var, -180, 180)
create_slider(sidebar, "Angle Min", angle_from_var, 0, 359)
create_slider(sidebar, "Angle Max", angle_to_var, 0, 359)

def tog_lbl(): global show_labels; show_labels = not show_labels
tk.Checkbutton(sidebar, text="Show Distance Labels", command=tog_lbl, bg=C_PANEL, fg="white", selectcolor="#333333", activebackground=C_PANEL).pack(anchor="w", pady=10)

status_lbl = tk.Label(sidebar, textvariable=var_status, bg=C_PANEL, fg=C_OK, font=("Arial", 10, "bold"))
status_lbl.pack(side="bottom", pady=10)

# ==========================================
#        MATPLOTLIB & INTERACTION
# ==========================================
plt.style.use('dark_background')
fig = plt.figure(facecolor='#000000', figsize=(5, 5))
ax = fig.add_subplot(111, polar=True)
ax.set_facecolor('black')
ax.grid(color='#222222', linewidth=0.5)
ax.spines['polar'].set_visible(False)

canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack(fill='both', expand=True)

def on_scroll(event):
    curr = range_var.get()
    if event.button == 'up': range_var.set(max(500, curr - 500))
    elif event.button == 'down': range_var.set(min(50000, curr + 500))
    save_settings()

def on_press(event):
    global last_mouse_x, last_mouse_y, is_dragging, zoom_start_pos
    if event.inaxes != ax: return
    
    if is_zoom_mode:
        if event.xdata is not None and event.ydata is not None:
            zoom_start_pos = (event.xdata, event.ydata)
    else:
        if event.button == 1: 
            is_dragging = True
            last_mouse_x = event.x
            last_mouse_y = event.y

def on_release(event): 
    global is_dragging, is_zoom_mode, zoom_start_pos
    
    if is_zoom_mode and zoom_start_pos and event.xdata is not None:
        start_theta, start_r = zoom_start_pos
        end_theta, end_r = event.xdata, event.ydata
        
        deg1 = np.rad2deg(start_theta) % 360
        deg2 = np.rad2deg(end_theta) % 360
        
        min_a = min(deg1, deg2)
        max_a = max(deg1, deg2)
        if abs(max_a - min_a) > 180: min_a, max_a = max_a, min_a
        
        new_max_r = max(start_r, end_r)
        
        angle_from_var.set(min_a)
        angle_to_var.set(max_a)
        range_var.set(new_max_r)
        
        zoom_start_pos = None
        toggle_zoom_mode() 
        save_settings()
        
    is_dragging = False

def on_drag(event):
    global last_mouse_x, last_mouse_y
    if is_dragging and not is_zoom_mode and event.x and event.y:
        dx = event.x - last_mouse_x
        dy = event.y - last_mouse_y
        if abs(dx) > abs(dy): rotation_var.set((rotation_var.get() + dx * 0.5) % 360)
        else: range_var.set(max(500, min(50000, range_var.get() + dy * 10)))
        last_mouse_x = event.x
        last_mouse_y = event.y

canvas.mpl_connect('scroll_event', on_scroll)
canvas.mpl_connect('button_press_event', on_press)
canvas.mpl_connect('button_release_event', on_release)
canvas.mpl_connect('motion_notify_event', on_drag)

# ==========================================
#             MAIN LOOP
# ==========================================
def update_gui():
    if is_closing: return
    
    # 1. Update Speed
    if (time.time() - last_pulse_time) > 3.0:
        global current_speed_kmph; current_speed_kmph = 0.0
    
    net_cycles = raw_hall_count - count_offset
    total_dist = net_cycles * DISTANCE_PER_CYCLE
    
    var_speed.set(f"{current_speed_kmph:.1f}")
    var_dist.set(f"{total_dist:.2f}")
    var_cycles.set(f"{net_cycles}")
    
    # 2. Collect Data
    r = range_var.get(); p = point_size_var.get()
    rot = rotation_var.get(); amin = angle_from_var.get(); amax = angle_to_var.get()

    angles = []; dists = []
    
    if plot_running:
        with data_lock:
            for ag_l, d_l in scan_history:
                for a, d in zip(ag_l, d_l):
                    ra = (a + rot) % 360
                    # Check visual window
                    if (amin <= amax and amin <= ra <= amax) or (amin > amax and (ra >= amin or ra <= amax)):
                        angles.append(np.deg2rad(ra)); dists.append(d)
        
        ax.clear()
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_ylim(0, r)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.grid(color='#222222', linewidth=0.5)
        
        if amin <= amax:
            ax.set_thetamin(amin)
            ax.set_thetamax(amax)

        if angles:
            ax.scatter(angles, dists, s=p, c=C_ACCENT, alpha=0.8, edgecolors='none')
            if show_labels:
                step = max(1, len(angles)//50)
                for i in range(0, len(angles), step):
                    ax.text(angles[i], dists[i], f"{int(dists[i])}", color="yellow", fontsize=7)
        
        canvas.draw_idle()

    # =====================================================
    #  CENTERED DEFLECTION DETECTION (0° = 178°)
    # =====================================================
    present_angles = set()
    for a in angles:
        deg = int(round(np.rad2deg(a) % 360))
        present_angles.add(deg)

    missing_angles = []
    
    start_check = 170
    end_check = 186
    CENTER_REF = 178 # This is our new "0"
    
    for deg in range(start_check, end_check + 1):
        if deg not in present_angles:
            missing_angles.append(deg)

    if missing_angles:
        ranges = []
        start = missing_angles[0]
        prev = start
        for d in missing_angles[1:]:
            if d == prev + 1:
                prev = d
            else:
                if prev-start >= 1: 
                    ranges.append((start,prev))
                start=d
                prev=d
        if prev-start >= 1:
            ranges.append((start,prev))

        if ranges:
            # We use carriage return (\r) so it doesn't spam new lines infinitely
            print(f"\n--- DEFLECTION (Center 0° = {CENTER_REF}°) ---")
            for rg in ranges:
                # Convert raw angles to centered relative angles
                rel_start = rg[0] - CENTER_REF
                rel_end   = rg[1] - CENTER_REF
                
                # Formatting sign
                s_start = f"{rel_start:+d}°"
                s_end   = f"{rel_end:+d}°"
                
                if rg[0]==rg[1]: 
                    print(f"  > Gap at: {s_start}")
                else: 
                    print(f"  > Deflection Zone: {s_start} to {s_end}")

    root.after(40, update_gui)

def update_sps():
    if is_closing: return
    global sps_counter
    var_sps.set(f"{sps_counter}")
    sps_counter = 0
    root.after(1000, update_sps)

def on_close():
    global is_closing; is_closing = True
    save_settings()
    root.destroy()
    sys.exit()

root.protocol("WM_DELETE_WINDOW", on_close)

# --- START ---
t = threading.Thread(target=serial_read_thread, daemon=True)
t.start()

root.after(100, update_gui)
root.after(1000, update_sps)
root.mainloop()
