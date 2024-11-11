from torch.utils.data import Dataset
from audio_preprocessing import AudioPreprocessing
import torchaudio
import torch


class SoundDataset(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 6000
        self.sr = 16000
        self.channel = 1
        self.shift_pct = 0.4
        self.noise = self.create_noise()


    def create_noise(self):
        SAMPLE_NOISE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        noise, _ = torchaudio.load(SAMPLE_NOISE)
        noise = torch.cat([noise for _ in range(20)], 1)
        return noise

    def __len__(self):
        return len(self.df) 


    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[idx, 'name']
        # Get the Class ID
        class_id = self.df.loc[idx, 'class_id']

        audio = torchaudio.load(audio_file)
        noise = self.noise[:, : audio[0].shape[1]]
        audio = (audio[0] + noise*10, audio[1])
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioPreprocessing.resample(audio, self.sr)
        #rechan = AudioPreprocessing.rechannel(reaud, self.channel)
        #audio = AudioPreprocessing.pad_trunc(rechan, self.duration)
        return reaud[0], class_id