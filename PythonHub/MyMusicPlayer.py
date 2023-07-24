import tkinter as tk
from tkinter import filedialog
import pygame
import os

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

# Initialize pygame mixer
pygame.mixer.init()

# GUI setup
root = tk.Tk()
root.title("Music Player")

# Tracks playlist
playlist_paths = []
playlist_frame = tk.Frame(root)
playlist_frame.pack(pady=10)

playlist_scroll = tk.Scrollbar(playlist_frame)
playlist_scroll.pack(side=tk.RIGHT, fill=tk.Y)

playlist = tk.Listbox(playlist_frame, selectbackground="gray", selectforeground="black", yscrollcommand=playlist_scroll.set, width=50)
playlist.pack()

playlist_scroll.config(command=playlist.yview)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

add_button = tk.Button(button_frame, text="Add Track", command=add_to_playlist)
add_button.grid(row=0, column=0, padx=5)

play_button = tk.Button(button_frame, text="Play", command=play_music)
play_button.grid(row=0, column=1, padx=5)

stop_button = tk.Button(button_frame, text="Stop", command=stop_music)
stop_button.grid(row=0, column=2, padx=5)

pause_button = tk.Button(button_frame, text="Pause", command=pause_music)
pause_button.grid(row=0, column=3, padx=5)

resume_button = tk.Button(button_frame, text="Resume", command=resume_music)
resume_button.grid(row=0, column=4, padx=5)

# Current playing track label
current_track = tk.StringVar()
current_track.set("")
current_track_label = tk.Label(root, textvariable=current_track)
current_track_label.pack(pady=20)

# Variable to keep track of paused state
is_paused = False

root.mainloop()

# C:\Users\janak\Desktop\GitHub\python\PythonHub