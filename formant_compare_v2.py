# formant_compare_v2.py
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy.signal import freqz, group_delay

# Compatibility shims for newer librosa
np.complex = complex
np.float = float

# ---------- CONFIG ----------
REAL_WAV = "8real.wav"          # <- change to your file
GEN_WAV  = "8fake.wav"     # <- change to your file
TARGET_SR = 16000
FRAME_MS = 25
HOP_MS   = 10
MAX_FORMANT_HZ = 5000
LPC_ORDER = 18                 # ~ 2 + sr/1000 for 16 kHz
CEP_LIFTER = 50                # quefrency lifter length (tune 30–70)
# ----------------------------

def load_and_preprocess(path):
    y, sr = librosa.load(path, sr=None)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    # pre-emphasis to flatten spectral tilt
    y = librosa.effects.preemphasis(y, coef=0.97)
    # normalize
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr

def frame_params(sr):
    frame_length = int(FRAME_MS * sr / 1000)
    hop_length   = int(HOP_MS   * sr / 1000)
    return frame_length, hop_length

def pick_voiced_frame(y, sr):
    """Pick a likely voiced vowel frame: high energy & low ZCR."""
    fl, hl = frame_params(sr)
    S = librosa.feature.rms(y=y, frame_length=fl, hop_length=hl).flatten()
    z = librosa.feature.zero_crossing_rate(y=y, frame_length=fl, hop_length=hl).flatten()
    idx_energy_top = np.argsort(S)[-10:]          # top-10 energy frames
    best = idx_energy_top[np.argmin(z[idx_energy_top])]
    start = best*hl
    frame = y[start:start+fl]
    if len(frame) < fl:                            # pad if last frame is short
        frame = np.pad(frame, (0, fl-len(frame)))
    return frame, best

def lpc_coeffs(frame, order):
    frame = frame * np.hamming(len(frame))
    A = librosa.lpc(y=frame, order=order)          # a[0] ≈ 1
    return A

def lpc_envelope(A, sr, n_fft=2048):
    # All-pole model H(z) = 1 / A(z)
    w, h = freqz(b=[1.0], a=A, worN=n_fft, fs=sr)
    env_db = 20*np.log10(np.maximum(np.abs(h), 1e-12))
    return w, env_db

def cepstral_envelope(frame, sr, n_fft=2048, lifter=CEP_LIFTER):
    mag = np.abs(np.fft.rfft(frame*np.hamming(len(frame)), n=n_fft))
    log_mag = np.log(np.maximum(mag, 1e-12))
    cep = np.fft.irfft(log_mag)
    # lifter: keep low-quefrency (spectral envelope), zero the rest
    cep_lift = np.copy(cep)
    cep_lift[lifter:-lifter] = 0
    env_log = np.fft.rfft(cep_lift).real
    env_db = 20*env_log
    freqs = np.linspace(0, sr/2, len(env_db))
    return freqs, env_db

def lpc_formants(A, sr):
    roots = np.roots(A)
    roots = roots[np.imag(roots) >= 0]
    angs = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angs * (sr/(2*np.pi))
    bw = - (sr/np.pi) * np.log(np.abs(roots))
    # plausible formants
    keep = (freqs > 90) & (freqs < MAX_FORMANT_HZ) & (bw < 400)
    f = np.sort(freqs[keep])
    return f

def track_formants(y, sr):
    fl, hl = frame_params(sr)
    frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    times = librosa.frames_to_time(np.arange(frames.shape[1]), sr=sr, hop_length=hl)
    f1=f2=f3=np.full(frames.shape[1], np.nan)
    for i in range(frames.shape[1]):
        A = lpc_coeffs(frames[:,i], LPC_ORDER)
        f = lpc_formants(A, sr)
        if len(f) > 0: f1[i] = f[0]
        if len(f) > 1: f2[i] = f[1]
        if len(f) > 2: f3[i] = f[2]
    return times, f1, f2, f3

def plot_all(y, sr, title_prefix, out_prefix):
    fl, hl = frame_params(sr)
    # (a) Spectrogram
    plt.figure(figsize=(9,4))
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=hl)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=hl, x_axis='time', y_axis='hz')
    plt.ylim(0, MAX_FORMANT_HZ)
    plt.title(f"(a) FFT Spectrogram – {title_prefix}")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_a_spectrogram.png", dpi=200)

    # Pick voiced frame
    frame, frame_idx = pick_voiced_frame(y, sr)
    A = lpc_coeffs(frame, LPC_ORDER)

    # (b) LPC Spectrum (envelope)
    plt.figure(figsize=(9,3))
    w, env_db = lpc_envelope(A, sr, n_fft=2048)
    plt.plot(w, env_db, linestyle="-", marker=".", markersize=2)
    plt.xlim(0, MAX_FORMANT_HZ)
    plt.xlabel("Hz"); plt.ylabel("dB")
    plt.title(f"(b) LPC Envelope – {title_prefix}")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_b_lpc_envelope.png", dpi=200)

    # (c) Cepstral/True Envelope
    plt.figure(figsize=(9,3))
    freqs, env_db2 = cepstral_envelope(frame, sr, n_fft=2048, lifter=CEP_LIFTER)
    plt.plot(freqs, env_db2, linestyle="-", marker=".", markersize=2)
    plt.xlim(0, MAX_FORMANT_HZ)
    plt.xlabel("Hz"); plt.ylabel("dB")
    plt.title(f"(c) Cepstral (True) Envelope – {title_prefix}")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_c_true_envelope.png", dpi=200)

    # (d) Group Delay (LPC)
    plt.figure(figsize=(9,3))
    w_gd, gd = group_delay((np.array([1.0]), A), 2048)
    freqs = np.linspace(0, sr/2, len(w_gd))   # scale to Hz manually
    plt.plot(freqs, gd, linestyle="-", marker=".", markersize=2)
    plt.xlim(0, MAX_FORMANT_HZ)
    plt.xlabel("Hz"); plt.ylabel("Group delay (samples)")
    plt.title(f"(d) Group Delay (from LPC) – {title_prefix}")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_d_group_delay.png", dpi=200)

    # (e/f) Spectrogram + Formant tracks
    plt.figure(figsize=(9,4))
    librosa.display.specshow(S, sr=sr, hop_length=hl, x_axis='time', y_axis='hz')
    plt.ylim(0, MAX_FORMANT_HZ)
    t, f1, f2, f3 = track_formants(y, sr)
    plt.scatter(t, f1, s=6, alpha=0.8, label="F1")
    plt.scatter(t, f2, s=6, alpha=0.8, label="F2")
    plt.scatter(t, f3, s=6, alpha=0.8, label="F3")
    plt.legend(loc="upper right", fontsize=8)
    plt.title(f"(e/f) Formant Tracks over Spectrogram – {title_prefix}")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_ef_formant_tracks.png", dpi=200)

def main():
    y_real, sr = load_and_preprocess(REAL_WAV)
    y_gen,  _  = load_and_preprocess(GEN_WAV)

    plot_all(y_real, sr, "Real", "real")
    plot_all(y_gen,  sr, "Generated", "gen")
    print("Saved: *_a_spectrogram.png, *_b_lpc_envelope.png, *_c_true_envelope.png, *_d_group_delay.png, *_ef_formant_tracks.png")

if __name__ == "__main__":
    main()
