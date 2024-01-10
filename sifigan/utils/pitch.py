# -*- coding: utf-8 -*-

import numpy as np
import parselmouth

"""Do F0 scaling.

References:
    - https://github.com/auspicious3000/contentvec

"""

def change_gender(x, fs, lo, hi, ratio_fs, ratio_ps, ratio_pr):
    s = parselmouth.Sound(x, sampling_frequency=fs)
    f0 = s.to_pitch_ac(pitch_floor=lo, pitch_ceiling=hi, time_step=0.8/lo)
    f0_np = f0.selected_array['frequency']
    f0_med = np.median(f0_np[f0_np!=0]).item()
    ss = parselmouth.praat.call([s, f0], "Change gender", ratio_fs, f0_med*ratio_ps, ratio_pr, 1.0)
    return ss.values.squeeze(0)

def change_gender_f0(x, fs, lo, hi, ratio_fs, new_f0_med, ratio_pr):
    s = parselmouth.Sound(x, sampling_frequency=fs)
    ss = parselmouth.praat.call(s, "Change gender", lo, hi, ratio_fs, new_f0_med, ratio_pr, 1.0)
    return ss.values.squeeze(0)

def random_formant_f0(wav, sr):
    # TODO: hard-coding
    np.random.seed(12345)
    rng = np.random.default_rng()
    lo, hi = 100, 400
    ratio_fs = rng.uniform(1, 1.4)
    coin = (rng.random() > 0.5)
    ratio_fs = coin*ratio_fs + (1-coin)*(1/ratio_fs)
    ratio_ps, ratio_pr = 1.0, 1.0
    
    ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)
    
    return ss, ratio_fs