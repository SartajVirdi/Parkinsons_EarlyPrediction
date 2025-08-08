import sounddevice as sd
from scipy.io.wavfile import write

# Sampling rate
fs = 44100  # 44.1 kHz is CD quality
seconds = 4   # Duration of recording

print("Recording started...")

# Record audio
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished

# Save as WAV file
write("test_audio1.wav", fs, recording)

print("Recording complete. File saved as test_audio.wav")
