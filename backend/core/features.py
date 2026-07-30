import mne
from mne.decoding import CSP


def extract_csp_features(epochs: mne.Epochs, labels: np.ndarray, n_components: int = 4):
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    features = csp.fit_transform(epochs.get_data(), labels)
    return features, csp

