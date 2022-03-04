# this code is copied directly from https://github.com/tianqi-zhou/mimic_imputation.
import numpy as np
from sklearn import preprocessing

SERIES_DIM=15

class Preprocessor():

    def __init__(self, _format='normal'):
        assert _format in {'normal', 'sequential'}, 'Unsupported data preprocessor'
        self.format = _format

        self.series_max = None
        self.series_min = None
        self.series_scaler = None

        self.adm_max = None
        self.adm_min = None
        self.adm_scaler = None

    def fit(self, data):
        """Fits self.series_scaler to the input data, and records the max and the min."""
        series_data = data

        # fitting scaler
        self.series_scaler = preprocessing.StandardScaler().fit(series_data)

        self.series_max = np.nanmax(series_data, axis=0, keepdims=True)
        self.series_min = np.nanmin(series_data, axis=0, keepdims=True)

    def preprocess(self, data):
        """Scales the input data with self.series_scaler."""
        assert self.series_scaler is not None

        series_data = data.copy()

        # transform with scaler
        series_data = self.series_scaler.transform(series_data)

        return series_data

    def postprocess_series(self, data):
        assert self.series_scaler is not None

        if self.format in {'normal'}:
            N = data.shape[0]
            series_data = data[:, :-ADM_DIM].copy()
            series_data = self.series_scaler.inverse_transform(series_data)
            series_data = series_data.clip(self.series_min, self.series_max)
        elif self.format in {'sequential'}:
            N, L = data.shape[:2]
            # series_data = data.copy().reshape([N*L, SERIES_DIM])
            series_data = data.copy().reshape([N, L*SERIES_DIM])
            series_data = self.series_scaler.inverse_transform(series_data)
            series_data = series_data.clip(self.series_min, self.series_max)
            series_data = series_data.reshape([N, L, SERIES_DIM])
        else:
            raise NotImplementedError('Oops...')

        return series_data