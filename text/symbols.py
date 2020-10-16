""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict
from .korean import ALL_SYMBOLS, PAD, EOS

_pad        = '_'
_sos        = '^'
_eos        = '~'
_punctuations = " ,.'?!"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
en_symbols = [_pad, _sos, _eos] + list(_punctuations) + list(_characters) + _arpabet
symbols = ALL_SYMBOLS

if __name__=="__main__":
    print(len(symbols))