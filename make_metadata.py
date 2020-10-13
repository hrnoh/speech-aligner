import argparse, os
import glob

def make_metadata(audio_path, txt_path, output_path, sep='|', use_spks=None):
    assert os.path.isdir(audio_path) and os.path.isdir(txt_path), "audio_path or txt_path is wrong"

    if use_spks is not None:
        all_spk_path = [p for p in glob.glob(os.path.join(audio_path, '*')) if os.path.isdir(p) and os.path.basename(p) in use_spks]
    else:
        all_spk_path = [p for p in glob.glob(os.path.join(audio_path, '*')) if os.path.isdir(p)]
    assert len(all_spk_path) > 0, "data does not exist."

    metadata = []
    for idx, spk_path in enumerate(all_spk_path):
        all_wav_path = glob.glob(os.path.join(spk_path, '*.wav'))
        assert len(all_wav_path) > 0, "{}'s data does not exist.".format(os.path.basename(spk_path))

        spk_name = os.path.basename(spk_path)
        for wav_path in all_wav_path:
            wav_name = os.path.basename(wav_path)
            audio_txt_path = os.path.join(txt_path, spk_name, wav_name[:-4] + ".txt")

            if not os.path.isfile(wav_path) or not os.path.isfile(audio_txt_path):
                print("{} does not exist".format(os.path.join(spk_name, wav_name)))
                continue

            with open(audio_txt_path, 'rt') as f:
                text = f.readline().rstrip()

            meta = sep.join([os.path.join(spk_name, wav_name), text, str(idx)])
            metadata.append(meta)

    metadata_str = "\n".join(metadata)
    with open(os.path.join(output_path, 'metadata.csv'), 'wt') as f:
        f.write(metadata_str)
    print("Done!")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--audio', type=str, default='data\\VCTK\\wav48')
    p.add_argument('--text', type=str, default='data\\VCTK\\txt')
    p.add_argument('--output', type=str, default='data\\VCTK\\')
    p.add_argument('--sep', type=str, default='|')
    p.add_argument('--use_spks', type=list, default=['p226', 'p301'])
    args = p.parse_args()

    make_metadata(args.audio, args.text, args.output, args.sep, args.use_spks)