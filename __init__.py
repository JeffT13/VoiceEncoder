name = "resemblyzer"

from ResemblyzeLegal.VoiceEncoder.audio import preprocess_wav, wav_to_mel_spectrogram, trim_long_silences, normalize_volume
from ResemblyzeLegal.VoiceEncoder.hparams import sampling_rate
from ResemblyzeLegal.VoiceEncoder.voice_encoder import VoiceEncoder
from ResemblyzeLegal.VoiceEncoder.util import case_to_dvec