"""
Functions for extracting filterbank and MFCC features.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from tqdm import tqdm
import glob
import numpy as np
import scipy.io.wavfile as wav


def extract_fbank_dir(dir):
    """
    Extract filterbanks for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension. Mel-scale log filterbanks are extracted.
    """
    import librosa
    feat_dict = {}
    for wav_fn in tqdm(sorted(glob.glob(path.join(dir, "*.wav")))):
        signal, sample_rate = librosa.core.load(wav_fn, sr=None)
        signal = preemphasis(signal, coeff=0.97)
        fbank = np.log(librosa.feature.melspectrogram(
            signal, sr=sample_rate, n_mels=40,
            n_fft=int(np.floor(0.025*sample_rate)),
            hop_length=int(np.floor(0.01*sample_rate)), fmin=64, fmax=8000,
            ))
        # from python_speech_features import logfbank
        # samplerate, signal = wav.read(wav_fn)
        # fbanks = logfbank(
        #     signal, samplerate=samplerate, winlen=0.025, winstep=0.01,
        #     nfilt=45, nfft=2048, lowfreq=0, highfreq=None, preemph=0,
        #     winfunc=np.hamming
        #     )
        key = path.splitext(path.split(wav_fn)[-1])[0]
        feat_dict[key] = fbank.T
    return feat_dict


def extract_mfcc_dir(dir):
    """
    Extract MFCCs for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension. Deltas and double deltas are also extracted.
    """
    import librosa
    feat_dict = {}
    for wav_fn in tqdm(sorted(glob.glob(path.join(dir, "*.wav")))):
        signal, sample_rate = librosa.core.load(wav_fn, sr=None)
        if len(signal) == 0:
            continue
        signal = preemphasis(signal, coeff=0.97)
        mfcc = librosa.feature.mfcc(
            signal, sr=sample_rate, n_mfcc=13, n_mels=24,  #dct_type=3,
            n_fft=int(np.floor(0.025*sample_rate)),
            hop_length=int(np.floor(0.01*sample_rate)), fmin=64, fmax=8000,
            #htk=True
            )
        # mfcc = librosa.feature.mfcc(
        #     signal, sr=sample_rate, n_mfcc=13,
        #     n_fft=int(np.floor(0.025*sample_rate)),
        #     hop_length=int(np.floor(0.01*sample_rate))
        #     )
        if mfcc.shape[1] < 9:  # need at least 9 frames for deltas
            continue
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        key = path.splitext(path.split(wav_fn)[-1])[0]
        feat_dict[key] = np.hstack([mfcc.T, mfcc_delta.T, mfcc_delta_delta.T])

        # from python_speech_features import delta
        # from python_speech_features import mfcc
        # sample_rate, signal = wav.read(wav_fn)
        # mfccs = mfcc(
        #     signal, samplerate=sample_rate, winlen=0.025, winstep=0.01,
        #     numcep=13, nfilt=24, nfft=None, lowfreq=0, highfreq=None,
        #     preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming
        #     )
        # d_mfccs = delta(mfccs, 2)
        # dd_mfccs = delta(d_mfccs, 2)
        # key = path.splitext(path.split(wav_fn)[-1])[0]
        # feat_dict[key] = np.hstack([mfccs, d_mfccs, dd_mfccs])

        # import matplotlib.pyplot as plt
        # plt.imshow(feat_dict[key][2000:2200,:])
        # plt.show()
        # assert False
    return feat_dict


def extract_vad(feat_dict, vad_dict):
    """
    Remove silence based on voice activity detection (VAD).

    The `vad_dict` should have the same keys as `feat_dict` with the active
    speech regions given as lists of tuples of (start, end) frame, with the end
    excluded.
    """
    output_dict = {}
    for utt_key in tqdm(sorted(feat_dict)):
        if utt_key not in vad_dict:
            print("Warning: Missing VAD for utterance", utt_key)
            continue
        for (start, end) in vad_dict[utt_key]:
            segment_key = utt_key + "_{:06d}-{:06d}".format(start, end)
            output_dict[segment_key] = feat_dict[utt_key][start:end, :]
    return output_dict


def speaker_mvn(feat_dict):
    """
    Perform per-speaker mean and variance normalisation.

    It is assumed that each of the keys in `feat_dict` starts with a speaker
    identifier followed by an underscore.
    """

    speakers = set([key.split("_")[0] for key in feat_dict])

    # Separate features per speaker
    speaker_features = {}
    for utt_key in sorted(feat_dict):
        speaker = utt_key.split("_")[0]
        if speaker not in speaker_features:
            speaker_features[speaker] = []
        speaker_features[speaker].append(feat_dict[utt_key])

    # Determine means and variances per speaker
    speaker_mean = {}
    speaker_std = {}
    for speaker in speakers:
        features = np.vstack(speaker_features[speaker])
        speaker_mean[speaker] = np.mean(features, axis=0)
        speaker_std[speaker] = np.std(features, axis=0)

    # Normalise per speaker
    output_dict = {}
    for utt_key in tqdm(sorted(feat_dict)):
        speaker = utt_key.split("_")[0]
        output_dict[utt_key] = (
            (feat_dict[utt_key] - speaker_mean[speaker]) / 
            speaker_std[speaker]
            )

    return output_dict


def preemphasis(signal, coeff=0.97):
    """Perform preemphasis on the input `signal`."""    
    return np.append(signal[0], signal[1:] - coeff*signal[:-1])
