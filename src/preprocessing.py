import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F
import soundfile as sf


class SpeechDataset(Dataset):
    def __init__(self, file_list, target_sample_rate: int = 16000, target_length: int = 16000, preload_in_memory: bool = False):
        self.file_list = file_list
        self.target_length = target_length
        self.target_sample_rate = target_sample_rate
        self.preload = preload_in_memory
        self.cache = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        if self.preload and idx in self.cache:
            return self.cache[idx]

        file_path, label = self.file_list[idx]

        # Load raw audio
        try:
            data, sample_rate = sf.read(file_path, dtype="float32")
            waveform = torch.from_numpy(data)
            waveform = waveform.unsqueeze(
                0) if waveform.dim() == 1 else waveform.t()
        except Exception:
            waveform, sample_rate = torchaudio.load(file_path)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            waveform = F.resample(waveform, orig_freq=sample_rate,
                                  new_freq=self.target_sample_rate)

        # Pad or truncate to normalize length
        length = waveform.shape[1]
        if length < self.target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.target_length - length))
        else:
            waveform = waveform[:, :self.target_length]

        result_waveform = waveform.squeeze(0)
        if self.preload:
            self.cache[idx] = (result_waveform, label)
        return result_waveform, label
