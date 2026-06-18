import wave
import struct
import math

sample_rate = 44100
duration = 2
frequency = 1000

wav_file = wave.open("alarm.wav.mp3", "w")
wav_file.setnchannels(1)
wav_file.setsampwidth(2)
wav_file.setframerate(sample_rate)

for i in range(int(sample_rate * duration)):
    value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
    data = struct.pack('<h', value)
    wav_file.writeframesraw(data)

wav_file.close()

print("alarm.wav.mp3 created!")