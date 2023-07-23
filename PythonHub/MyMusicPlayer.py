import pygame

def play_music(file_path):
    pygame.init()
    pygame.mixer.init()

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pass

    except pygame.error:
        print("Error loading or playing the music.")

    pygame.mixer.music.stop()
    pygame.quit()

if __name__ == "__main__":
    music_file = "C:\\Users\\janak\\Desktop\\GitHub\\python\\PythonHub\\file.mp3"  # Replace with the path to your music file
    play_music(music_file)
# C:\Users\janak\Desktop\GitHub\python\PythonHub