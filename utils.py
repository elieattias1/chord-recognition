import numpy as np
import librosa


def get_major_templates():
    "templates for major 7 chords"
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
    return arr / norms


def get_minor_templates():
    "templates for major 7 chords"
    arr = np.array(
        [  # C, C# D, D# E, F, F# G, G# A, A# B
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # C
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # C#
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # D
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # D#
            [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # E
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],  # F
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],  # F#
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],  # G
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],  # G#
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # A
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # A#
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],  # B
        ]
    )

    norms = np.linalg.norm(arr, axis=0)
    return arr / norms


def get_chord_templates(chord_type="major"):
    if chord_type == "major":
        return get_major_templates()
    else:
        return get_minor_templates()


def get_match(templates, chroma_vector):
    chords = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    similarities = templates @ chroma_vector
    # print(similarities)
    match = np.argmax(similarities)
    return chords[match]


def normalize_chroma(chroma):
    return chroma / np.linalg.norm(chroma)


def recognize_chord(y, sr, templates):
    if y.max() - y.min() < 0.01:
        return "silence", np.zeros((12, 1))
    chroma_cq = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
    )
    mean_chroma = np.mean(chroma_cq, axis=1)
    mean_chroma = normalize_chroma(mean_chroma)

    match = get_match(templates, mean_chroma)
    return match, chroma_cq
