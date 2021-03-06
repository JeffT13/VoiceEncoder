
import numpy as np
import torch, json

from pathlib import Path
from typing import Union, List, Optional
from torch import nn
from time import perf_counter as timer
from collections import Counter
from math import floor, ceil

from VoiceEncoder.hparams import *
from VoiceEncoder.util import *
from VoiceEncoder import audio


class VoiceEncoder(nn.Module):
    def __init__(self, device: Union[str, torch.device]=None, verbose=True, weights_fpath: Union[Path, str]=None):
        """
        If None, defaults to cuda if it is available on your machine, otherwise the model will
        run on cpu. Outputs are always returned on the cpu, as numpy arrays.
        :param weights_fpath: path to "<CUSTOM_MODEL>.pt" file path.
        If None, defaults to built-in "pretrained.pt" model
        """
        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Load the pretrained model'speaker weights
        if weights_fpath is None:
            weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
        else:
            weights_fpath = Path(weights_fpath)

        if not weights_fpath.exists():
            raise Exception("Couldn't find the voice encoder pretrained model at %s." %
                            weights_fpath)
        start = timer()
        checkpoint = torch.load(weights_fpath, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

        if verbose:
            print("Loaded the voice encoder model on %s in %.2f seconds." %
                  (device.type, timer() - start))

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size).
        Embeddings are positive and L2-normed, thus they lay in the range [0, 1].
        """
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to
        obtain partial utterances of <partials_n_frames> each. Both the waveform and the
        mel spectrogram slices are returned, so as to make each partial utterance waveform
        correspond to its spectrogram.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wav_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        assert 0 < min_coverage <= 1

        # Compute how many frames separate two partial utterances
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        assert 0 < frame_step, "The rate is too high"
        assert frame_step <= partials_n_frames, "The rate is too low, it should be %f at least" % \
            (sampling_rate / (samples_per_frame * partials_n_frames))

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self, wav: np.ndarray, mask: np.ndarray, wav_labels: Optional[np.ndarray]=None, sd_path: Optional[str]=None, rate=4, min_coverage=.75, overlap = .1, spkr_thres=.75, verbose=False):
        """
        Computes an embedding for a single utterance. The utterance is divided in partial
        utterances and an embedding is computed for each. The complete utterance embedding is the
        L2-normed average embedding of the partial utterances.
        TODO: independent batched version of this function
        :param wav: a preprocessed utterance waveform as a numpy array of float32
        :param return_partials: if True, the partial embeddings will also be returned along with
        the wav slices corresponding to each partial utterance.
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
        returned.
        """
        # Compute where to split the utterance into partials and pad the waveform with zeros if
        # the partial utterances cover a larger range.
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
            if wav_labels is not None:
                wav_labels = np.pad(wav_labels, (0, max_wave_length - len(wav_labels)), "constant", constant_values=998)

        # Split the utterance into partials
        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        
        if wav_labels is not None:
            with open(sd_path) as json_file: 
              spkr_dict = json.load(json_file)
            mel_lab = audio.wav_label_for_melspec(wav, wav_labels)
            # -----------
            # Handle dvectors with overlapped labels
            lab_lst = []
            tracker = 0
            for k, s in enumerate(mel_slices):
              if len(np.unique(mel_lab[s]))==1:
                lab_lst.append(mel_lab[s][0])
              else:
                data = Counter(mel_lab[s])
                if data.most_common()[0][1]/len(mel_lab[s])>spkr_thres: #if 1 speaker is 75% of utterance label as spkr
                  lab_lst.append(data.most_common()[0][0])
                else:
                  lab_lst.append(997) #label as shared dvec
                  tracker+=1
            if verbose:
              print("num of (mainly) cross spkr windows:", tracker)
            PE_labels = np.asarray(lab_lst)
            # -----------
            
        # -----
        # Build embedding audio time labels 
        ms = 1/16
        wav2time = np.asarray([ms*i for i in range(len(mask))])
        w = wav2time[mask==True] # take times that were marked as audio
        timetrack = np.array([(w[s][0], w[s][-1]) for s in wav_slices]) # take start and end times for dvec label
        # -----

        # -----------------
        # GPU Memory Management (w/ overlap)
        # forward mels thru model
        cut_div = ceil(2*rate)
        cut = np.shape(mels)[0]//cut_div
        t0 = 0
        temp_emb = []
        te = []

        with torch.no_grad():
          for i in range(cut_div):
            if i<(cut_div-1):
              mel_samp = mels[t0:t0+cut, :, :]
            else:
              mel_samp = mels[t0:, :, :]

            #process slice
            mel_samp = torch.from_numpy(mel_samp).to(self.device)
            emb  = self(mel_samp).cpu().numpy()
            temp_emb.append(emb)
            t0+=floor(cut*(1-overlap))

        #consolidate temp_emb
        for i in range(cut_div):
          if i==0:
            te.append(temp_emb[i])
          else:
            te.append(temp_emb[i][ceil(cut*overlap):])
        # -----------------

        partial_embeds = np.concatenate(te, axis=0)
        
        if wav_labels is not None:
            results = (partial_embeds, PE_labels, timetrack)
            info = (wav_slices, mel_slices, mel)
            return results, info
        else:
            return partial_embeds, (timetrack, wav_slices)
            

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
        """
        Compute the embedding of a collection of wavs (presumably from the same speaker) by
        averaging their embedding and L2-normalizing it.

        :param wavs: list of wavs a numpy arrays of float32.
        :param kwargs: extra arguments to embed_utterance()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,).
        """
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) \
                             for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)
