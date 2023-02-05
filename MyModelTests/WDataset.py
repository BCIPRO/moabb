import mne
import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from scipy.io import loadmat, savemat
from sklearn.pipeline import make_pipeline

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset

sfreq = 250
num_session = 10
channels = 8
strim = 10 * sfreq
slabs = 5 * sfreq
slen = 60 * sfreq
data_path = "../my_data/" # location of data

def store_subj_data(subj):
    #labels = np.array(([1] * slabs + [2] * slabs) * (slen // (slabs * 2)), dtype=int)
    mdict = {"fs":sfreq}

    for i in range(num_session):
        n = data_path + "s{}_{}.bdf".format(subj, i)
        # only the first 8 are channels, and we remove the buffer
        end = slen + strim
        data = mne.io.read_raw_bdf(n, verbose=10000).get_data()[:channels, strim : end]
        #data_with_labels = np.vstack([data, labels])

        filename = "subject_" + str(subj + 1).zfill(2) + ".mat"
        
        mdict["x" + str(i)] = data
        
    savemat(filename, mdict)

    return data


# Create the fake data
for subject in [0]:
    store_subj_data(subject)

class WenData(BaseDataset):
    """
    Dataset used to exemplify the creation of a dataset class in MOABB.
    The data samples have been simulated and has no physiological meaning
    whatsoever.
    """

    def __init__(self):
        super().__init__(
            subjects=[1],
            sessions_per_subject=num_session,
            events={"open": 1, "closed": 2},
            code="Wen dataset",
            interval=[0, 0.75],
            paradigm="imagery",
            doi="",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path_list = self.data_path(subject)

        data = loadmat(file_path_list[0])
        fs = data["fs"]
        sessions = {}

        annot = mne.Annotations(np.arange(0, 60, 5), 5, np.array(["open", "closed"] * 6))
        
        for sess in range(num_session):
            x = data["x" + str(sess)]
            ch_names = ["ch" + str(i) for i in range(8)]
            ch_types = ["eeg" for i in range(8)]
            info = mne.create_info(ch_names, fs, ch_types)
            raw = mne.io.RawArray(x, info)
            raw.set_annotations(annot)
            #raw.plot()
            sessions["session_" + str(sess)] = {}
            sessions["session_" + str(sess)]["run_1"] = raw
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from one subject"""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        return ["./subject_01.mat"]
