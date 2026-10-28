import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Ensure compatibility with librosa
np.complex = complex
np.float = float

# === LOAD AUDIO FILES ===
real_path = "8real.wav"        # replace with your filename
gen_path = "8fake.wav"    # replace with your filename

real_audio, real_sr = librosa.load(real_path, sr=None)
gen_audio, gen_sr = librosa.load(gen_path, sr=None)

# === HELPER FUNCTIONS ===
def lpc_spectrum(signal, order=16, sr=16000):
    """Compute LPC spectral estimate of signal"""
    signal = signal * np.hamming(len(signal))
    A = librosa.lpc(y=signal, order=order)   # ✅ use keyword arguments
    freqs = np.linspace(0, sr/2, 512)
    return freqs, -20*np.log10(np.abs(np.fft.fft(A, 1024)[:512]))

def formant_estimate(signal, sr, order=16):
    """Estimate formant frequencies from LPC roots"""
    A = librosa.lpc(y=signal, order=order)   # ✅ use keyword arguments
    roots = np.roots(A)
    roots = [r for r in roots if np.imag(r) >= 0]
    angs = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angs * (sr / (2*np.pi))
    return sorted(freqs)

# Use a short segment for LPC & formants
segment_real = real_audio[:2048]
segment_gen = gen_audio[:2048]

# === CREATE FIGURE ===
fig, axes = plt.subplots(6, 2, figsize=(14, 20))

# (a) FFT Spectrum
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(real_audio)), ref=np.max),
                         sr=real_sr, x_axis='time', y_axis='hz', ax=axes[0,0])
axes[0,0].set_title("(a) FFT Spectrum - Real")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(gen_audio)), ref=np.max),
                         sr=gen_sr, x_axis='time', y_axis='hz', ax=axes[0,1])
axes[0,1].set_title("(a) FFT Spectrum - Generated")

# (b) LPC Spectrum
freqs, spec_real = lpc_spectrum(segment_real, order=16, sr=real_sr)
freqs, spec_gen = lpc_spectrum(segment_gen, order=16, sr=gen_sr)
axes[1,0].plot(freqs, spec_real)
axes[1,0].set_title("(b) LPC Spectrum - Real")
axes[1,1].plot(freqs, spec_gen)
axes[1,1].set_title("(b) LPC Spectrum - Generated")

# (c) Cepstral smoothing (MFCC features)
mfcc_real = librosa.feature.mfcc(y=real_audio, sr=real_sr, n_mfcc=13)
mfcc_gen = librosa.feature.mfcc(y=gen_audio, sr=gen_sr, n_mfcc=13)
librosa.display.specshow(mfcc_real, x_axis='time', ax=axes[2,0])
axes[2,0].set_title("(c) Cepstral Features - Real")
librosa.display.specshow(mfcc_gen, x_axis='time', ax=axes[2,1])
axes[2,1].set_title("(c) Cepstral Features - Generated")

# (d) True Envelope approximation
S_real = np.abs(librosa.stft(real_audio))
S_gen = np.abs(librosa.stft(gen_audio))
env_real = librosa.decompose.nn_filter(S_real, aggregate=np.median, metric='cosine')
env_gen = librosa.decompose.nn_filter(S_gen, aggregate=np.median, metric='cosine')
librosa.display.specshow(librosa.amplitude_to_db(env_real, ref=np.max), sr=real_sr, x_axis='time', y_axis='hz', ax=axes[3,0])
axes[3,0].set_title("(d) True Envelope Approx - Real")
librosa.display.specshow(librosa.amplitude_to_db(env_gen, ref=np.max), sr=gen_sr, x_axis='time', y_axis='hz', ax=axes[3,1])
axes[3,1].set_title("(d) True Envelope Approx - Generated")

# (e) Group delay approximation
spec_real = np.fft.fft(real_audio[:4096])
spec_gen = np.fft.fft(gen_audio[:4096])
gd_real = -np.diff(np.unwrap(np.angle(spec_real)))
gd_gen = -np.diff(np.unwrap(np.angle(spec_gen)))
axes[4,0].plot(gd_real)
axes[4,0].set_title("(e) Group Delay Approx - Real")
axes[4,1].plot(gd_gen)
axes[4,1].set_title("(e) Group Delay Approx - Generated")

# (f) Formant estimate
formants_real = formant_estimate(segment_real, real_sr)
formants_gen = formant_estimate(segment_gen, gen_sr)
axes[5,0].stem(formants_real, [1]*len(formants_real), basefmt=" ")
axes[5,0].set_xlim(0, 5000)
axes[5,0].set_title("(f) Formant Estimates - Real")
axes[5,1].stem(formants_gen, [1]*len(formants_gen), basefmt=" ")
axes[5,1].set_xlim(0, 5000)
axes[5,1].set_title("(f) Formant Estimates - Generated")

plt.tight_layout()
plt.show()
