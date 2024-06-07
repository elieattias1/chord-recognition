import argparse
import numpy as np
import pyaudio
import wave
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import get_major_templates, recognize_chord, get_chord_templates


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Determine the type of chord (major or minor)"
    )
    parser.add_argument(
        "--chord_type",
        default="all",
        type=str,
        help="Specify if you wish to recognize major, minor or all types of chords",
    )
    if not os.path.exists("recorded"):
        os.makedirs("recorded")
    args = parser.parse_args()
    chord_type = args.chord_type
    return chord_type


def initialize_audio_stream(channels, rate, chunk):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )
    return p, stream


def record_audio(stream, wf, rate, chunk, record_seconds, templates, chords):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(0, 2 * chunk, 2)
    res = []
    (line,) = ax.plot(x, [0] * len(x), c="b")
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, chunk)
    ax.set_ylabel("Displacement")
    ax.set_xlabel("Chunk Size")
    title = f"Real Time Audio Waveform\nDetecting Chords"
    ax.set_title(title)

    print("Recording...")

    def update(frame):
        chunk_data = stream.read(chunk, exception_on_overflow=False)
        int_data = np.frombuffer(chunk_data, dtype=np.int16) / 32768.0
        chord, chroma_cq = recognize_chord(int_data, rate, templates, chords)
        wf.writeframes(chunk_data)
        line.set_ydata(int_data)
        # add_on = "m" if chord_type == "minor" else ""
        _title = "silence" if chord == "silence" else f"chord: {chord}"
        ax.set_title(title + " " + _title)

        return line, ax

    ani = FuncAnimation(fig, update, blit=False)
    plt.show()

    print("Done")


def main():
    chord_type = parse_arguments()
    print(f"Detecting {chord_type} chords")

    CHUNK = 1024 * 8  # Increased buffer size
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 if sys.platform == "darwin" else 2
    RATE = 44100
    RECORD_SECONDS = 50

    p, stream = initialize_audio_stream(CHANNELS, RATE, CHUNK)

    templates, chords = get_chord_templates(chord_type=chord_type)
    print(chords)
    with wave.open("recorded/output.wav", "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        record_audio(stream, wf, RATE, CHUNK, RECORD_SECONDS, templates, chords)

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
