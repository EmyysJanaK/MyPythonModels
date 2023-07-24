import tkinter as tk
from tkinter import filedialog
import pygame
import os

# Colors
BG_COLOR = "#EFEFEF"
BUTTON_COLOR = "#4CAF50"
BUTTON_TEXT_COLOR = "white"
PROGRESS_COLOR = "#2196F3"

def play_music():
    global is_paused
    if not pygame.mixer.music.get_busy():
        if playlist.curselection():
            selected_track = playlist.curselection()[0]
            pygame.mixer.music.load(playlist_paths[selected_track])
            pygame.mixer.music.play()
            current_track.set(os.path.basename(playlist_paths[selected_track]))
            is_paused = False

def stop_music():
    pygame.mixer.music.stop()
    current_track.set("")

def pause_music():
    global is_paused
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
        is_paused = True

def resume_music():
    global is_paused
    if is_paused:
        pygame.mixer.music.unpause()
        is_paused = False

def add_to_playlist():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav;*.ogg")])
    if file_path:
        playlist_paths.append(file_path)
        playlist.insert(tk.END, os.path.basename(file_path))

def on_playlist_select(event):
    selected_track = playlist.curselection()
    if selected_track:
        track_info = pygame.mixer.Sound(playlist_paths[selected_track[0]]).get_length()
        current_track_info.set(f"Track: {os.path.basename(playlist_paths[selected_track[0]])}\nDuration: {int(track_info // 60)}m {int(track_info % 60)}s")

# Initialize pygame mixer
pygame.mixer.init()

# GUI setup
root = tk.Tk()
root.title("Music Player")
root.config(bg=BG_COLOR)

# Tracks playlist
playlist_paths = []
playlist_frame = tk.Frame(root, bg=BG_COLOR)
playlist_frame.pack(pady=10)

playlist_scroll = tk.Scrollbar(playlist_frame)
playlist_scroll.pack(side=tk.RIGHT, fill=tk.Y)

playlist = tk.Listbox(playlist_frame, selectbackground="gray", selectforeground="black", yscrollcommand=playlist_scroll.set, width=50, bg=BG_COLOR)
playlist.pack()

playlist_scroll.config(command=playlist.yview)
playlist.bind('<<ListboxSelect>>', on_playlist_select)

# Buttons
button_frame = tk.Frame(root, bg=BG_COLOR)
button_frame.pack(pady=10)

add_button = tk.Button(button_frame, text="Add Track", command=add_to_playlist, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR)
add_button.grid(row=0, column=0, padx=5)

play_button = tk.Button(button_frame, text="Play", command=play_music, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR)
play_button.grid(row=0, column=1, padx=5)

stop_button = tk.Button(button_frame, text="Stop", command=stop_music, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR)
stop_button.grid(row=0, column=2, padx=5)

pause_button = tk.Button(button_frame, text="Pause", command=pause_music, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR)
pause_button.grid(row=0, column=3, padx=5)

resume_button = tk.Button(button_frame, text="Resume", command=resume_music, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR)
resume_button.grid(row=0, column=4, padx=5)

# Current playing track label
current_track = tk.StringVar()
current_track.set("")
current_track_label = tk.Label(root, textvariable=current_track, bg=BG_COLOR)
current_track_label.pack(pady=5)

# Current playing track info label
current_track_info = tk.StringVar()
current_track_info.set("")
current_track_info_label = tk.Label(root, textvariable=current_track_info, bg=BG_COLOR)
current_track_info_label.pack(pady=5)

# Progress bar
def update_progress():
    if pygame.mixer.music.get_busy():
        current_time = pygame.mixer.music.get_pos() // 1000
        track_info = pygame.mixer.Sound(playlist_paths[playlist.curselection()[0]]).get_length()
        progress = int((current_time / track_info) * 100)
        progress_bar['value'] = progress
    root.after(1000, update_progress)

progress_bar = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, length=300, bg=BG_COLOR, fg=PROGRESS_COLOR, troughcolor=BG_COLOR, highlightthickness=0, sliderrelief=tk.FLAT)
progress_bar.pack(pady=10)

# Volume slider
def set_volume(val):
    volume = int(val) / 100
    pygame.mixer.music.set_volume(volume)

volume_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume, bg=BG_COLOR)
volume_slider.set(50)  # Default volume value
volume_slider.pack(pady=5)

# Variable to keep track of paused state
is_paused = False

root.after(1000, update_progress)
root.mainloop()
