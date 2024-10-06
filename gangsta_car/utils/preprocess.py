import numpy as np

def preprocess(waveform, norm_percentile):
    """
    Normalize the waveform from -1 to 1 using max abs normalization.

    waveform : 1d input tensor
    norm_percentile: float to not consider the topmost values in case of an error during normalization
    """
    # Normalize waveforms
    max_abs = np.percentile(np.abs(waveform), float(norm_percentile)*100)
    waveform = waveform / max_abs

    # Clip values to be within the range [-1, 1]
    waveform = np.clip(waveform, -1, 1)
    return waveform