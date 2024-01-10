import os
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf
import pysptk
import pyworld
from sifigan.utils import read_txt

# A parameter called "all-pass constant" which depends on sampling rate (16kHz: 0.42, 22.5kHz: 0.457 , 24kHz: 0.466)
alpha = {16000: 0.42, 22500: 0.457, 24000: 0.466}


class ObjectiveScore:
    def __init__(
        self,
        sr=24000,  # Sampling rate
        n_fft=2048,  # FFT size
        frame_period=10.0,  # Hop size in [ms]
        f0_floor=100.0,  # Minimum F0 for F0 estimation [Hz]
        f0_ceil=1000.0,  # Maximum F0 for F0 estimation [Hz]
        n_mels=40,  # Number of bins for mel-cepstram extraction [Hz]
        exclude_0th_component_in_LSD=True,
        exclude_0th_component_in_MCD=True,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.frame_period = frame_period
        self.hop_size = int(sr * frame_period * 0.001)
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.n_mels = n_mels
        self.alpha = alpha[sr]
        self.exclude_0th_component_in_LSD = exclude_0th_component_in_LSD
        self.exclude_0th_component_in_MCD = exclude_0th_component_in_MCD

        # For calculating F0-RMSE, LSD and MCD
        self.n_frames = [0 for _ in range(3)]
        self.means = [0.0 for _ in range(3)]

        # For calculating V/UV decision error
        self.voiced_threshold = 0  # Threshold used to decide V/UV
        self.n_uv_error_frame = 0  # Initialize number of V/UV errored frames
        self.n_all_frame = 0  # Initialize number of all frames

    def online_mean(self, n, mean, x):
        for i in range(x.shape[0]):
            new_mean = (n * mean + x[i]) / (n + 1)
            mean = new_mean
            n += 1
        return n, new_mean

    def frame(self, x, center=False):
        # framing function
        if center:
            pad = np.zeros(self.n_fft // 2)
            frames = librosa.util.frame(
                np.concatenate((pad, x)), self.n_fft, self.hop_size, axis=0
            )
        else:
            frames = librosa.util.frame(x, self.n_fft, self.hop_size, axis=0)
        return frames

    def analyze_f0(self, x, f0_factor=1):
        # return F0 sequence
        _f0, t = pyworld.dio(
            x.astype(np.float64),
            fs=self.sr,
            f0_floor=self.f0_floor * f0_factor,
            f0_ceil=self.f0_ceil * f0_factor,
            frame_period=self.frame_period,
        )
        f0 = pyworld.stonemask(x.astype(np.float64), _f0, t, self.sr)
        return f0

    def analyze_mcep(self, x):
        # return mel-cepstrum
        frames = self.frame(x)
        frames *= pysptk.hanning(self.n_fft)
        mcep = pysptk.mcep(frames, self.n_mels, self.alpha)
        if self.exclude_0th_component_in_MCD:
            mcep = mcep[:, 1:]
        return mcep

    def analyze_spec(self, x):
        # return log power spectrum
        S = librosa.core.stft(x, n_fft=self.n_fft, hop_length=self.hop_size).T
        S_db = 10 * np.log10(
            np.maximum(
                np.abs(S) ** 2,
                1e-7,
            )
        )
        if self.exclude_0th_component_in_LSD:
            ceps = np.fft.ifft(S_db)
            ceps[:, 0] = 0
            S_db = np.fft.fft(ceps).real
        return S_db

    def calc_f0_mse(self, x, y):
        # return mean squared error of F0
        x_voiced = x > self.voiced_threshold
        y_voiced = y > self.voiced_threshold
        xy_voiced = x_voiced & y_voiced
        f0_mse = (np.log(x[xy_voiced]) - np.log(y[xy_voiced])) ** 2
        return f0_mse[:, np.newaxis]  # (T, 1)

    def calc_uv_dicision_error(self, x, y):
        # return V/UV dicision error
        x_voiced = x > self.voiced_threshold
        y_voiced = y > self.voiced_threshold
        uv_dicision_error = x_voiced ^ y_voiced
        return uv_dicision_error[:, np.newaxis]

    def calc_mcep_distortion(self, x, y):
        # return mel-cepstrum distortion
        x_mc = self.analyze_mcep(x)
        y_mc = self.analyze_mcep(y)
        mcd = 10 / np.log(10) * ((2 * np.sum((x_mc - y_mc) ** 2, axis=1)) ** 0.5)
        # print(x_mc.shape, mcd.shape, "<- must be (T, D), (T, )")
        return mcd[:, np.newaxis]  # (T, 1)

    def calc_log_spec_distortion(self, x, y):
        # return log power spectrum distortion
        x_S_db = self.analyze_spec(x)
        y_S_db = self.analyze_spec(y)
        diff_S_db = (x_S_db - y_S_db) ** 2
        lsd = np.mean(diff_S_db, axis=1) ** 0.5
        # print(diff_S_db.shape, lsd.shape, "<- must be (T, D) and (T,)")
        return lsd[:, np.newaxis]  # (T, 1)

    def append_data(self, x, y, f0_factor=1.0):
        # make input signals 1D
        x = x.reshape(max(x.shape))
        y = y.reshape(max(y.shape))

        # adjust length
        if x.shape[0] > y.shape[0]:
            diff = x.shape[0] - y.shape[0]
            right = diff // 2
            left = diff - right
            x = x[left:-right]  # (T,)
        elif x.shape[0] < y.shape[0]:
            diff = y.shape[0] - x.shape[0]
            left = diff // 2
            right = diff - left
            y = y[left:-right]  # (T,)

        # analyse F0
        x_f0 = self.analyze_f0(x, f0_factor)
        y_f0 = self.analyze_f0(y) * f0_factor

        # analyse V/UV
        x_voiced = x_f0 > self.voiced_threshold
        y_voiced = y_f0 > self.voiced_threshold
        uv_dicision_error = x_voiced ^ y_voiced
        self.n_uv_error_frame += sum(uv_dicision_error)
        self.n_all_frame += uv_dicision_error.shape[0]

        # calculate scores
        scores = [
            self.calc_f0_mse(x_f0, y_f0),
            self.calc_mcep_distortion(x, y),
            self.calc_log_spec_distortion(x, y),
        ]

        # calculate new means of scores
        for i in range(len(scores)):
            if scores[i].shape[0] > 0:
                self.n_frames[i], self.means[i] = self.online_mean(
                    self.n_frames[i], self.means[i], scores[i]
                )


def main():
    # settings
    sample_rate = 24000
    method_dir = "exp/sifigan/wav/500000"
    f0_factor = 1.0
    print(f"Method: {method_dir}")
    print(f"F0 factor: {f0_factor}")

    # NOTE: f0_floor and f0_ceil effect the accuracy of F0 estimation
    # so it is recommended to decide these parameters carefully.
    obj_score = ObjectiveScore(
        sr=sample_rate,
        n_fft=2048,
        f0_floor=100.0,  # should be adjusted if f0_factor < 1.0
        f0_ceil=1000.0,  # should be adjusted if f0_factor > 1.0
    )

    # online mean calculation
    files = read_txt("data/scp/namine_ritsu_v2_eval.scp")
    file_cnt = 0
    for wavname in tqdm(files):
        # read natural speech
        x, sr = sf.read(wavname)
        assert sr == sample_rate
        # read synthesized speech
        fname = wavname.replace("data/wav", method_dir)
        if f0_factor != 1.0:
            fname = fname.replace(".wav", f"_f{f0_factor:.2f}.wav")
        y, sr = sf.read(fname)
        assert sr == sample_rate
        # online calculation
        obj_score.append_data(x, y, f0_factor)
        file_cnt += 1
    print(f"Processed {file_cnt} files.")

    # show results
    means = obj_score.means
    print(f"RMSE of F0: {means[0][0] ** 0.5}")  # [Hz]
    print(
        f"V/UV Error: {100 * obj_score.n_uv_error_frame / obj_score.n_all_frame}"
    )  # [%]
    print(f"MCD: {means[1][0]}")  # [dB]
    #if f0_factor == 1:
    print(f"LSD: {means[2][0]}")  # [dB]


if __name__ == "__main__":
    main()