import os
import torch
import pickle
import operator
import logging
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from statistics import NormalDist
from scipy.stats import gamma
from sklearn.mixture import GaussianMixture
from functools import partial
from itertools import compress
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


def sliding_window_sequences(X, index, y=None, window_size=100, step_size=1, target_size=1):
    if y is None:
        y = X.copy()

    windows = list()
    targets = list()
    indices = list()

    length = len(X)
    for i in range(0, length - window_size - 1, step_size):
        start = i
        end = i + window_size
        windows.append(X[start: end])
        targets.append(y[end + 1])
        indices.append(index[end + 1])

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(indices)


class Model(torch.nn.Module):
    def __init__(self, seq_len, in_channels, out_channels, lstm_units=80, n_layer=1, dropout=0.2):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lstm_units = lstm_units
        self.lstm1 = torch.nn.LSTM(
            input_size=self.in_channels,
            hidden_size=self.lstm_units,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=self.lstm_units,
            hidden_size=self.lstm_units,
            batch_first=True
        )
        self.dense = torch.nn.Linear(self.lstm_units, self.out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.seq_len, self.in_channels))
        x, (_, _) = self.lstm1(x)
        x, (hidden_n, _) = self.lstm2(x)
        hidden_n = hidden_n.reshape((batch_size, self.lstm_units))
        out = self.dense(hidden_n)
        return out


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.best_score = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta or validation_loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True

        else:
            self.counter = 0

        if validation_loss < self.best_score:
            self.best_score = validation_loss


class LSTM:
    """LSTM Forecasting Model."""

    def __init__(self, out_channels=None, lstm_units=80, n_layer=1, dropout=0.2, device='cuda',
                 batch_size=32, lr=1e-3, verbose=True):
        self.out_channels = out_channels
        self.lstm_units = lstm_units
        self.n_layer = n_layer
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        # infer
        self.seq_len = None
        self.n_channels = None

    def fit(self, X, y, validation_split=0.2, epochs=35, tolerance=5, min_delta=10, checkpoint=50, path='checkpoint'):
        validation_size = int(len(X) * validation_split)
        train, train_y = torch.Tensor(X[:-validation_size]), torch.Tensor(y[:-validation_size])
        valid, valid_y = torch.Tensor(X[-validation_size:]), torch.Tensor(y[-validation_size:])

        # get shape
        _, self.seq_len, self.n_channels = X.shape
        if self.out_channels is None:
            self.out_channels = y.shape[1]

        train_loader = DataLoader(TensorDataset(train, train_y), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid, valid_y), batch_size=self.batch_size)

        self.model = Model(
            self.seq_len,
            self.n_channels,
            self.out_channels,
            self.lstm_units,
            self.n_layer,
            self.dropout
        ).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.history = dict(train=[], valid=[])

        early_stopping = EarlyStopping(tolerance=tolerance, min_delta=min_delta)

        for epoch in range(epochs):
            self.model.train()

            train_losses = []
            valid_losses = []

            for (x, y) in train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()

            for i, (x, y) in enumerate(valid_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, y)
                valid_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            valid_loss = np.mean(valid_losses)

            if self.verbose:
                print(f'Epoch {epoch + 1}/{epochs}: train loss {train_loss: .4f} | val loss {valid_loss: .4f}.')

            self.history['train'].append(train_loss)
            self.history['valid'].append(valid_loss)

            if epoch % checkpoint == 0:
                os.makedirs(path, exist_ok=True)
                file = os.path.join(path, f'model-{epoch}.pth')
                torch.save(self.model.state_dict(), file)

            early_stopping(train_loss, valid_loss)
            if early_stopping.early_stop:
                if self.verbose:
                    print(f'Early stopping at epoch: {epoch + 1}')
                break

    def predict(self, X):
        test_loader = DataLoader(dataset=X, batch_size=1)

        output = list()
        self.model.eval()
        for x in test_loader:
            x = x.to(self.device)
            pred = self.model(x)
            pred = pred.squeeze().to('cpu').detach().numpy()
            output.append(pred)

        return np.array(output)


LOGGER = logging.Logger(__file__)


def _get_sum(name, sensors):
    return sum([1 for sensor in sensors if name in sensor])


def _divide(x, y):
    return x / y if y else 0


def _get_default(sensors):
    return {sensor: 1 for sensor in sensors}


def _find_weights(sensors, prefix=None) -> dict:
    prefix = prefix or _get_default(sensors)

    pre_weights = {
        sensor_type: _divide(sensor_weight, _get_sum(sensor_type, sensors))
        for sensor_type, sensor_weight in prefix.items()
    }

    weights = [pre_weights[k] for sensor in sensors for k in pre_weights if k in sensor]

    return weights


def _compute_cdf(gmm, x):
    means = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_.flatten()
    cdf = 0
    for i in range(len(means)):
        cdf += weights[i] * NormalDist(mu=means[i], sigma=sigma[i]).cdf(x)

    return cdf


def _combine_pval(cdf, side=True):
    if side:
        p_val = 1 - cdf
    else:
        p_val = 2 * np.array(list(map(np.min, zip(1 - cdf, cdf))))

    p_val[p_val < 1e-16] = 1e-16

    fisher_pval = - 2 * np.log(p_val)

    return fisher_pval, p_val


class GMM:
    """Gaussian Mixture Model."""

    def _parse_components(self, n_components, sensors, default=1):
        if sensors is None:
            if isinstance(n_components, dict):
                raise ValueError("Unknown list of sensors but specified in components.")

            elif isinstance(n_components, int):
                return n_components

            return default

        if isinstance(n_components, dict):
            n_components = {**n_components,
                            **{k: default for k in sensors if k not in n_components}
                            }

        elif isinstance(n_components, int):
            n_components = dict(zip(sensors, [n_components] * len(sensors)))

        return n_components

    def __init__(self, sensors, n_components=1, covariance_type='spherical', one_sided=False, weights=None):
        self.sensors = sensors
        self.n_components = self._parse_components(n_components, sensors)
        self.covariance_type = covariance_type
        self.one_sided = one_sided

        self.gmm = [None] * len(self.sensors)
        self.compute_cdf = np.vectorize(_compute_cdf)

        self.g_scale = None
        self.g_shape = None

        self.weights = weights or _find_weights(self.sensors)

    def fit(self, X):
        combined = 0
        num_sensors = X.shape[1]
        assert num_sensors == len(self.sensors)
        for i, sensor in enumerate(self.sensors):
            x = X[:, i].reshape(-1, 1)
            gmm = GaussianMixture(n_components=self.n_components[sensor],
                                  covariance_type=self.covariance_type)

            gmm.fit(x)
            self.gmm[i] = gmm

            cdf = self.compute_cdf(gmm, x.flatten())
            fisher, p_val = _combine_pval(cdf, self.one_sided)

            combined += self.weights[i] * fisher

        if np.var(combined) > 0:
            self.g_scale = np.var(combined) / np.mean(combined)
            self.g_shape = np.mean(combined) ** 2 / np.var(combined)

        else:
            LOGGER.warning(f'No variance found between p-values ({np.var(combined)}).')
            self.g_scale = 1
            self.g_shape = 0

    def p_values(self, X):
        combined = 0
        p_val_sensors = np.zeros_like(X)
        fisher_values = np.zeros_like(X)
        for i, sensor in enumerate(self.sensors):
            y = X[:, i]
            gmm = self.gmm[i]
            cdf = self.compute_cdf(gmm, y)

            fisher, p_val = _combine_pval(cdf, self.one_sided)
            combined += self.weights[i] * fisher

            p_val_sensors[:, i] = p_val
            fisher_values[:, i] = fisher

        gamma_p_val = 1 - gamma.cdf(combined, a=self.g_shape, scale=self.g_scale)

        return gamma_p_val, p_val_sensors, combined, fisher_values


def _smooth(errors, smoothing_window):
    smoothed_errors = pd.DataFrame(errors).ewm(smoothing_window).mean().values

    return smoothed_errors


def point_errors(y, pred, smooth=False, smoothing_window=10):
    errors = np.abs(y - pred)

    if smooth:
        errors = _smooth(errors, smoothing_window)

    errors = np.array(errors)

    return errors


def area_errors(y, pred, score_window=10, dx=100, smooth=False, smoothing_window=10):
    trapz = partial(np.trapz, dx=dx)
    errors = np.empty_like(y)
    num_signals = errors.shape[1]

    for i in range(num_signals):
        area_y = pd.Series(y[:, i]).rolling(
            score_window, center=True, min_periods=score_window // 2).apply(trapz)
        area_pred = pd.Series(pred[:, i]).rolling(
            score_window, center=True, min_periods=score_window // 2).apply(trapz)

        error = area_y - area_pred

        if smooth:
            error = _smooth(error, smoothing_window)

        errors[:, i] = error.flatten()

    mu = np.mean(errors)
    std = np.std(errors)

    return (errors - mu) / std


def _merge_sequences(sequences):
    if len(sequences) == 0:
        return np.array([])

    sorted_sequences = sorted(sequences, key=operator.itemgetter(0))
    new_sequences = [sorted_sequences[0]]
    score = [sorted_sequences[0][2]]

    for sequence in sorted_sequences[1:]:
        prev_sequence = new_sequences[-1]

        if sequence[0] <= prev_sequence[1] + 1:
            score.append(sequence[2])
            average = np.mean(score)
            new_sequences[-1] = (prev_sequence[0], max(prev_sequence[1], sequence[1]), average)
        else:
            score = [sequence[2]]
            new_sequences.append(sequence)

    return np.array(new_sequences)


class M2AD:
    def __init__(self, dataset, entity, sensors=None, covariates=None, time_column='time',
                 feature_range=(0, 1), strategy='mean', error_name='point',
                 window_size=100, target_size=1, step_size=1, lstm_units=80, n_layer=2,
                 dropout=0.2, device='cpu', batch_size=32, lr=1e-3, epochs=35, score_window=10,
                 verbose=True, n_components=1, covariance_type='spherical', **kwargs):
        self.dataset = dataset
        self.entity = entity
        self.sensors = sensors
        self.covariates = covariates
        self.time_column = time_column

        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.imputer = SimpleImputer(strategy=strategy)

        self.window_size = window_size
        self.target_size = target_size
        self.step_size = step_size

        self.model = LSTM(
            lstm_units=lstm_units,
            n_layer=n_layer,
            dropout=dropout,
            device=device,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose
        )

        self.error_name = error_name
        self.gamma_thresh = 0.001

        self.epochs = epochs
        self.score_window = score_window

        self.n_components = n_components
        self.covariance_type = covariance_type

        if self.error_name == 'point':
            self.one_sided = True
        elif self.error_name == 'area':
            self.one_sided = False
        else:
            raise ValueError(f"Unknown error function {self.error_name}.")

    def _get_data(self, df):
        data = df.copy()
        timestamp = data.pop(self.time_column)

        if self.covariates:
            X = data[self.sensors]
            cov = data[self.covariates]

        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        y = X.copy()

        X = np.concatenate([X, cov.values], axis=1) if self.covariates else X

        return X, y, cov, timestamp

    def _compute_error(self, y, pred):
        if self.error_name == 'point':
            return point_errors(y, pred, smooth=True)
        elif self.error_name == 'area':
            return area_errors(y, pred, smooth=True)

        raise ValueError(f"Unknown error function {self.error_name}.")

    def create_intervals(self, anomalies, index, score, anomaly_padding=50):
        intervals = list()
        length = len(anomalies)
        anomalies_index = list(compress(range(length), anomalies))
        for idx in anomalies_index:
            start = max(0, idx - anomaly_padding)
            end = min(idx + anomaly_padding + 1, length)
            value = np.mean(score[start: end])
            intervals.append([index[start], index[end], value])

        intervals = _merge_sequences(intervals)
        intervals = sorted(intervals, key=operator.itemgetter(0), reverse=True)

        anomalies = pd.DataFrame(intervals, columns=['start', 'end', 'score'])
        anomalies.insert(0, 'dataset', self.dataset)
        anomalies.insert(1, 'entity', self.entity)

        return anomalies

    def fit(self, df, validation_split=0.2, tolerance=10, min_delta=10):
        X, y, cov, timestamp = self._get_data(df)

        assert len(cov.columns) == len(self.covariates), f'check covariates {cov.columns}'
        assert y.shape[1] == len(self.sensors), f'check sensors {y.shape}'
        assert X.shape[1] == (len(self.covariates) + len(self.sensors)), f'check input {X.shape}'

        windows, targets, indices = sliding_window_sequences(
            X=X,
            y=y,
            index=timestamp,
            window_size=self.window_size,
            target_size=self.target_size,
            step_size=self.step_size
        )

        LOGGER.info(f'Training the model with {self.epochs} epochs.')
        self.model.fit(windows, targets, epochs=self.epochs, validation_split=validation_split,
                       tolerance=tolerance, min_delta=min_delta)

        pred = self.model.predict(windows)
        pred = pred.reshape(targets.shape)

        errors = self._compute_error(targets, pred)

        self.gmm_model = GMM(sensors=self.sensors,
                             n_components=self.n_components,
                             covariance_type=self.covariance_type,
                             one_sided=self.one_sided)

        self.gmm_model.fit(errors)
        anomalyscore, pval, fisher, _ = self.gmm_model.p_values(errors)
        self.threshold = np.percentile(fisher, 99.5)

        self.train_mse = mean_squared_error(targets, pred)
        self.train_errors = errors
        self.train_targets = targets
        self.train_pred = pred
        self.train_pvals = pval
        self.train_anomalyscore = anomalyscore

    def detect(self, df, debug=False):
        X, y, _, timestamp = self._get_data(df)
        windows, targets, indices = sliding_window_sequences(
            X=X,
            y=y,
            index=timestamp,
            window_size=self.window_size,
            target_size=self.target_size,
            step_size=self.step_size
        )

        pred = self.model.predict(windows)
        pred = pred.reshape(targets.shape)
        errors = self._compute_error(targets, pred)

        LOGGER.info(f'Applying threshold on p-values.')
        anomalyscore, pval, fisher, fisher_values = self.gmm_model.p_values(errors)
        anomalies = anomalyscore < self.gamma_thresh
        formatted_anomalies = self.create_intervals(anomalies, indices, anomalyscore)

        if debug:
            visuals = {
                "anomalies": anomalies,
                "test_targets": targets,
                "test_pred": pred,
                "test_errors": errors,
                "test_pvals": pval,
                "test_anomaly_score": anomalyscore,
                "test_timestamps": indices,
                "train_targets": self.train_targets,
                "train_pred": self.train_pred,
                "train_errors": self.train_errors,
                "train_pvals": self.train_pvals,
                "train_anomaly_score": self.train_anomalyscore,
            }

            return formatted_anomalies, visuals

        return formatted_anomalies




