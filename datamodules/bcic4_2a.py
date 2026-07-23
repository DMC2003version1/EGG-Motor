
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4
from sklearn.model_selection import train_test_split
import os


def _ordered_session_items(splitted_ds):
    def sort_key(item):
        key, _ = item
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        return (int(digits) if digits else 999, str(key))

    return sorted(splitted_ds.items(), key=sort_key)


def _get_2a_train_test_sessions(windows_dataset):
    splitted_ds = windows_dataset.split("session")
    if "session_T" in splitted_ds and "session_E" in splitted_ds:
        return splitted_ds["session_T"], splitted_ds["session_E"]

    keys = list(splitted_ds.keys())
    train_keys = [key for key in keys if "train" in str(key).lower()]
    test_keys = [
        key for key in keys
        if "test" in str(key).lower() or "eval" in str(key).lower()
    ]
    if train_keys and test_keys:
        return splitted_ds[train_keys[0]], splitted_ds[test_keys[0]]

    ordered_sessions = [dataset for _, dataset in _ordered_session_items(splitted_ds)]
    if len(ordered_sessions) < 2:
        raise KeyError(f"Expected at least 2 BCIC IV-2a sessions, got {keys}")
    return ordered_sessions[0], ordered_sessions[1]


class BCICIV2a(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4 
    
    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        train_dataset, test_dataset = _get_2a_train_test_sessions(self.dataset)

        # load the data
        X, y = BaseDataModule._dataset_to_arrays(train_dataset)
        X_test, y_test = BaseDataModule._dataset_to_arrays(test_dataset)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)                                                                
        # self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, 
                                                                #  preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test, 
                                                                #  preprocessing_dict=self.preprocessing_dict, mode="test")


class BCICIV2aTVT(BaseDataModule):
    val_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4 

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        # Split by session
        session1, session2 = _get_2a_train_test_sessions(self.dataset)
        
        # Load session 1 data
        X, y = BaseDataModule._dataset_to_arrays(session1)

        # Split session 1: 80% train, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.preprocessing_dict.get("seed", 42), stratify=y)

        # Load session 2 as test set
        X_test, y_test = BaseDataModule._dataset_to_arrays(session2)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        # Create datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        # self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val, y_val, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="val")
        # self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="test")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        #   persistent_workers=True,          # ↩︎ keeps workers alive between epochs
                        #   prefetch_factor=4                 # ↩︎ each worker preloads 4 future batches                          
                        )


class BCICIV2aLOSO(BCICIV2a):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=self.all_subject_ids, dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        train_datasets = [
            _get_2a_train_test_sessions(splitted_ds[str(subj_id)])[0]
            for subj_id in train_subjects
        ]
        val_datasets = [
            _get_2a_train_test_sessions(splitted_ds[str(subj_id)])[1]
            for subj_id in train_subjects
        ]
        test_dataset = _get_2a_train_test_sessions(splitted_ds[str(self.subject_id)])[1]

        # load the data
        train_arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in train_datasets]
        val_arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in val_datasets]
        X = np.concatenate([arr[0] for arr in train_arrays], axis=0)
        y = np.concatenate([arr[1] for arr in train_arrays], axis=0)
        X_val = np.concatenate([arr[0] for arr in val_arrays], axis=0)
        y_val = np.concatenate([arr[1] for arr in val_arrays], axis=0)
        X_test, y_test = BaseDataModule._dataset_to_arrays(test_dataset)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

        # Drop raw MOABB windows after materializing tensors (avoids 2x System RAM in LOSO)
        self.dataset = None
        del train_arrays, val_arrays, train_datasets, val_datasets, test_dataset
        del X, y, X_val, y_val, X_test, y_test, splitted_ds

        # self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val, y_val, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="val")
        # self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="test")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self._dataloader_kwargs(shuffle=False))
