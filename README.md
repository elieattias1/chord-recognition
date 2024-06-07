# Chord Recognition

This project implements a real-time chord recognition system using audio input. The system uses `librosa` for audio processing and `pyaudio` for real-time audio capture. The recognized chords are displayed along with a real-time waveform and chromatogram visualization. 


## Features

- Real-time chord recognition ([major](https://youtu.be/vzidvwVdLBQ ) and minor chords). 
- Real-time audio waveform visualization
- Real-time chromatogram visualization

## Installation

### Requirements

- Python 3.6+
- `numpy`
- `pyaudio`
- `wave`
- `librosa`
- `matplotlib`
- `argparse`

### Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/elieattias1/chord-recognition.git
    cd chord-recognition
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the script with the following command to recognize major chords:

```sh
python main.py --major
