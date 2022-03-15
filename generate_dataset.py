import os
import yaml
import mirdata
import librosa
import argparse
import numpy as np


def generate_examples(split, sample_rate, dataset_name, val_prop, mono):
    """Generate full track audio - annotation examples
    Args:
        split (str): data split. One of 'train' or 'validation'

    Yields:
        (np.ndarray, mirdata.annotations.F0Data): audio, annotation tuple
    """
    medleydb_pitch = mirdata.initialize(dataset_name)
    split_track_ids = medleydb_pitch.get_random_track_splits([1. - val_prop, val_prop])
    if split == "train":
        track_ids = split_track_ids[0]
    else:
        track_ids = split_track_ids[1]

    total_track_ids = len(track_ids)
    for i, track_id in enumerate(track_ids):
        print(f"Track {i+1}/{total_track_ids}: {track_id}")
        track = medleydb_pitch.track(track_id)
        y, _ = librosa.load(track.audio_path, sr=sample_rate, mono=mono)
        yield (y, track.pitch)


def generate_samples_from_example(audio, pitch_data, sample_rate, hop_length, cqt_frequencies, n_time_frames, n_example_per_track, bins_per_octave, n_octaves, harmonics, fmin):
    """Generate training samples given an audio - annotation tuple

    Args:
        audio (np.ndarray): full length audio signal (sr=22050)
        pitch_data (mirdata.annotations.F0Data): f0 data object

    Yields:
        (np.ndarray, np.ndarray, int): harmonic CQT, target salience and slice index
    """
    hcqt = compute_hcqt(audio, sample_rate=sample_rate, hop_length=hop_length, bins_per_octave=bins_per_octave, n_octaves=n_octaves, harmonics=harmonics, fmin=fmin)
    n_times = hcqt.shape[1]
    times = librosa.frames_to_time(np.arange(n_times), sr=sample_rate, hop_length=hop_length)

    target_salience = pitch_data.to_matrix(times, "s", cqt_frequencies, "hz")

    all_time_indexes = np.arange(0, n_times - n_time_frames)
    time_indexes = np.random.choice(all_time_indexes, size=(n_example_per_track), replace=False)

    # split example into time slices
    for t_idx in time_indexes:
        sample = hcqt[:, t_idx : t_idx + n_time_frames]
        label = target_salience[t_idx : t_idx + n_time_frames, :, np.newaxis]
        yield (sample, label, t_idx)

def compute_hcqt(audio, sample_rate, hop_length, bins_per_octave, n_octaves, harmonics, fmin):
    cqt_list = list()
    shapes = list()
    
    for h in harmonics:
        cqt = librosa.cqt(
            audio,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin * float(h),
            n_bins=bins_per_octave * n_octaves
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)
    
    if not all([s == shapes[0] for s in shapes]):
        min_time = np.min(shape[1] for shape in shapes)
        new_cqt_list = list()
        for cqt in cqt_list:
            new_cqt_list.append(cqt[:, :min_time])
        cqt_list = new_cqt_list
    
    log_hcqt = ((1. / 80.) * librosa.core.amplitude_to_db(np.abs(np.array(cqt_list)), ref=np.max)) + 1
    
    return log_hcqt


def main(args):

    # create data save directories if they don't exist
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    with open(args.cfg_path, 'r') as file:
        CFG = yaml.load(file, Loader=yaml.FullLoader)
    
    DATASET_NAME = CFG["dataset_name"]
    SAMPLE_RATE = CFG["sample_rate"]
    MONO = CFG["mono"]
    N_BINS = CFG["n_bins"]
    FMIN = CFG["fmin"]
    BINS_PER_OCTAVE = CFG["bins_per_octave"]
    N_OCTAVES = CFG["n_octaves"]
    HARMONICS = CFG["harmonics"]
    CQT_FREQUENCIES = librosa.cqt_frequencies(N_BINS, FMIN, BINS_PER_OCTAVE)
    HOP_LENGTH = CFG["hop_length"]
    N_TIME_FRAMES = CFG["n_time_frames"]
    N_EXAMPLES_PER_TRACK = CFG["n_examples_per_track"]
    VAL_PROPORTION = CFG["val_proportion"]
    assert 0. < VAL_PROPORTION < 1., f"Valdation proportion is not in [0, 1], found {VAL_PROPORTION}"

    splits = ["train", "validation"]
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)

    # initialize split-wise generators
    generators = [generate_examples(split, sample_rate=SAMPLE_RATE, dataset_name=DATASET_NAME, val_prop=VAL_PROPORTION, mono=MONO) for split in splits]

    # generate training and validation data
    for generator, split in zip(generators, splits):
        print(f"Generating {split} data...")
        for i, (audio, pitch_data) in enumerate(generator):
            for hcqt, target_salience, t_idx in generate_samples_from_example(
                audio,
                pitch_data,
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                cqt_frequencies=CQT_FREQUENCIES,
                n_time_frames=N_TIME_FRAMES,
                n_example_per_track=N_EXAMPLES_PER_TRACK,
                bins_per_octave=BINS_PER_OCTAVE,
                n_octaves=N_OCTAVES,
                harmonics=HARMONICS,
                fmin=FMIN
            ):
                fname = os.path.join(data_dir, split, f"{i}-{t_idx}.npz")
                np.savez(fname, hcqt=hcqt, target_salience=target_salience)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate npz files for training/validation")
    parser.add_argument("--data-dir", type=str, help="Directory to save the data in")
    parser.add_argument("--cfg-path", type=str, default="dataset_cfg.yaml", help="Path to the config file")
    args = parser.parse_args()
    main(args)
