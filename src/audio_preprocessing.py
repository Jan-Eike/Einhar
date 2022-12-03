import torch
import torchaudio
import random

class AudioPreprocessing():    
    @staticmethod
    def rechannel(audio, new_channel):
        signal, sampling_rate = audio

        if (signal.shape[0] == new_channel):
            return audio

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = signal[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([signal, signal])
        return ((resig, sampling_rate))


    @staticmethod
    def resample(audio, new_signal_rate):
        signal, sampling_rate = audio

        if (sampling_rate == new_signal_rate):
            return audio

        num_channels = signal.shape[0]
        # Resample first channel
        resampled_signal = torchaudio.transforms.Resample(sampling_rate, new_signal_rate)(signal[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sampling_rate, new_signal_rate)(signal[1:,:])
            resampled_signal = torch.cat([resampled_signal, retwo])

        return ((resampled_signal, new_signal_rate))

    
    @staticmethod
    def pad_trunc(audio, max_ms):
        signal, sampling_rate = audio
        num_rows, signal_len = signal.shape
        max_len = sampling_rate//1000 * max_ms

        if (signal_len > max_len):
            # Truncate the signal to the given length
            signal = signal[:,:max_len]

        elif (signal_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - signal_len)
            pad_end_len = max_len - signal_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            signal = torch.cat((pad_begin, signal, pad_end), 1)
        
        return (signal, sampling_rate)

    
    @staticmethod
    def time_shift(audio, shift_limit):
        signal, sampling_rate = audio
        _, signal_len = signal.shape
        shift_amt = int(random.random() * shift_limit * signal_len)
        return (signal.roll(shift_amt), sampling_rate)

    
    @staticmethod
    def specgram(audio, n_mels=64, n_fft=1024, hop_len=None, top_db=80):
        signal, sampling_rate = audio

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        specgram = torchaudio.transforms.MelSpectrogram(sampling_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

        # Convert to decibels
        specgram = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(specgram)
        return (specgram)


    @staticmethod
    def spectro_augment(specgram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = specgram.shape
        mask_value = specgram.mean()
        aug_specgram = specgram

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_specgram = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_specgram, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_specgram = torchaudio.transforms.TimeMasking(time_mask_param)(aug_specgram, mask_value)

        return aug_specgram

    
    @staticmethod
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)
        