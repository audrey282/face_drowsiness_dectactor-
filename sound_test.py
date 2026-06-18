import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav.mp3")
pygame.mixer.music.play()

time.sleep(10)