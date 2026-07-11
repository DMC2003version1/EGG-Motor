from typing import Dict

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)


def scale(data, factor):
    """Version-independent array scaling for Braindecode preprocessors."""
    return data * factor


def load_bcic4(subject_ids: list, dataset: str = "2a", preprocessing_dict: Dict = None,
              verbose: str = "WARNING"):
    # MOABB renamed these datasets in newer releases.  Accept both names so
    # the same branch runs with current Colab and the pinned legacy stack.
    dataset_names = ("BNCI2014_001", "BNCI2014001") if dataset == "2a" else (
        "BNCI2014_004", "BNCI2014004"
    )
    last_error = None
    for dataset_name in dataset_names:
        try:
            dataset = MOABBDataset(dataset_name, subject_ids=subject_ids)
            break
        except ValueError as exc:
            last_error = exc
    else:
        raise last_error

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose)
    ]

    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        preprocessors.append(Preprocessor("filter", l_freq=l_freq, h_freq=h_freq,
                                          verbose=verbose))

    preprocess(dataset, preprocessors)

    # create windows
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples, preload=False
    )

    return windows_dataset
