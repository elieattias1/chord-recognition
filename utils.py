import numpy as np
import librosa


def get_major_templates():
    "templates for major 7 chords"
    chords = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    arr = np.array(
        [  # C, C# D, D# E, F, F# G, G# A, A# B
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # C
            [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C#
            [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D#
            [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],  # E
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # F
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],  # F#
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],  # G
            [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],  # G#
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],  # A
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],  # A#
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],  # B
        ]
    )

    norms = np.linalg.norm(arr, axis=0)
    return arr / norms, chords


def get_minor_templates():
    "templates for major 7 chords"
    chords = [
        "Cm",
        "C#m",
        "Dm",
        "D#m",
        "Em",
        "Fm",
        "F#m",
        "Gm",
        "G#m",
        "Am",
        "A#m",
        "Bm",
    ]
    arr = np.array(
        [
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # Cm
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # C#m
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # Dm
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # D#m
            [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # Em
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],  # Fm
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],  # F#m
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],  # Gm
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],  # G#m
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # Am
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # A#m
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],  # Bm
        ]
    )

    norms = np.linalg.norm(arr, axis=0)
    return arr / norms, chords


def get_chord_templates(chord_type="major"):
    if chord_type == "major":
        return get_major_templates()
    elif chord_type == "minor":
        return get_minor_templates()
    else:
        major, chords_major = get_major_templates()
        minor, chords_minor = get_minor_templates()
        all_chords = chords_major + chords_minor
        return np.concatenate((major, minor), axis=0), all_chords


def get_match(templates, chroma_vector, chords):
    similarities = templates @ chroma_vector
    # print(similarities)
    match = np.argmax(similarities)
    return chords[match]


def normalize_chroma(chroma):
    return chroma / np.linalg.norm(chroma)


def recognize_chord(y, sr, templates, chords):
    if y.max() - y.min() < 0.01:
        return "silence", np.zeros((12, 1))
    chroma_cq = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
    )
    mean_chroma = np.mean(chroma_cq, axis=1)
    mean_chroma = normalize_chroma(mean_chroma)

    match = get_match(templates, mean_chroma, chords)
    return match, chroma_cq
