import matplotlib.pyplot as plt
import torch


class AudioUtils():
    @staticmethod
    def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
        waveform = waveform.numpy()
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate
        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show(block=False)


    @staticmethod
    def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
        waveform = waveform.numpy()
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate
        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)


    @staticmethod
    def print_stats(waveform, sample_rate=None, src=None):
        if src:
            print("-" * 10)
            print("Source:", src)
            print("-" * 10)
        if sample_rate:
            print("Sample Rate:", sample_rate)
        print("Shape:", tuple(waveform.shape))
        print("Dtype:", waveform.dtype)
        print(f" - Max:     {waveform.max().item():6.3f}")
        print(f" - Min:     {waveform.min().item():6.3f}")
        print(f" - Mean:    {waveform.mean().item():6.3f}")
        print(f" - Std Dev: {waveform.std().item():6.3f}")
        print()
        print(waveform)
        print()
        