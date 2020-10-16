import os
import librosa
from librosa.filters import mel as librosa_mel_fn
import pickle as pkl
import torch
import numpy as np
import codecs
import matplotlib.pyplot as plt
from tqdm import tqdm

from g2pk import G2p
from text import *
from text import cmudict
from text.cleaners import custom_english_cleaners
from text.symbols import symbols

from layers import TacotronSTFT

symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}

csv_file = '/hd0/speech-aligner/metadata/metadata.csv'
root_dir = '/hd0/dataset/VCTK/VCTK-Corpus/wav48'
data_dir = '/hd0/speech-aligner/preprocessed/VCTK20_engspks'

os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, 'char_seq'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'phone_seq'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'melspectrogram'), exist_ok=True)

g2p = G2p()
metadata = {}
with codecs.open(csv_file, 'r', 'utf-8') as fid:
    for line in fid.readlines():
        id, text, spk = line.split("|")
        id = os.path.splitext(id)[0]

        clean_char = custom_english_cleaners(text.rstrip())
        clean_phone = []
        for s in g2p(clean_char.lower()):
            if '@' + s in symbol_to_id:
                clean_phone.append('@' + s)
            else:
                clean_phone.append(s)

        metadata[id] = {'char': clean_char,
                        'phone': clean_phone}

stft = TacotronSTFT(filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=16000, mel_fmin=55.0,
                 mel_fmax=7600.0)

def text2seq(text):
    sequence = [symbol_to_id['^']]
    sequence.extend([symbol_to_id[c] for c in text])
    sequence.append(symbol_to_id['~'])
    return sequence


def get_mel(filename):
    wav, sr = librosa.load(filename, sr=16000)
    wav = librosa.effects.trim(wav, top_db=20, frame_length=1024, hop_length=256)[0]

    wav = torch.FloatTensor(wav.astype(np.float32))
    melspec = stft.mel_spectrogram(wav.unsqueeze(0))
    return melspec.squeeze(0).numpy(), wav


def save_file(fname):
    wav_name = os.path.join(root_dir, fname) + '.wav'
    text = metadata[fname]['char']
    char_seq = np.asarray(text2seq(metadata[fname]['char']), dtype=np.int64)
    try:
        phone_seq = np.asarray(text2seq(metadata[fname]['phone']), dtype=np.int64)
    except:
        phone_seq = np.asarray(text2seq([phone.replace('..', '.') for phone in metadata[fname]['phone']]),
                               dtype=np.int64)

    melspec, wav = get_mel(wav_name)

    # Skip existing files
    if os.path.isfile(os.path.join(data_dir, 'char_seq', f'{fname}_sequence.npy')) and \
            os.path.isfile(os.path.join(data_dir, 'phone_seq', f'{fname}_sequence.npy')) and \
            os.path.isfile(os.path.join(data_dir, 'melspectrogram', f'{fname}_melspectrogram.npy')):
        return text, char_seq, phone_seq, melspec, wav

    spk_name = fname[:4]
    os.makedirs(os.path.join(data_dir, 'char_seq', spk_name), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'phone_seq', spk_name), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'melspectrogram', spk_name), exist_ok=True)

    np.save(os.path.join(data_dir, 'char_seq', f'{fname}_sequence.npy'), char_seq)
    np.save(os.path.join(data_dir, 'phone_seq', f'{fname}_sequence.npy'), phone_seq)
    np.save(os.path.join(data_dir, 'melspectrogram', f'{fname}_melspectrogram.npy'), melspec)

    return text, char_seq, phone_seq, melspec, wav

mel_values = []
for k in tqdm(metadata.keys()):
    text, char_seq, phone_seq, melspec, wav = save_file(k)
    mel_values.extend(list(melspec.reshape(-1)))
    if k == 'p226/p226_001':
        print("Text:")
        print(text)
        print()
        print("Melspectrogram:")
        plt.figure(figsize=(16,4))
        plt.imshow(melspec, aspect='auto', origin='lower')
        plt.show()

mel_values = np.asarray(mel_values)
plt.hist(mel_values[:100000], bins=100)