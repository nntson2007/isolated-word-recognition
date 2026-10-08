import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelExtractor(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160, n_mels: int = 40):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            center=False,
            n_mels=n_mels,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.dim() == 3 and waveforms.size(1) == 1:
            waveforms = waveforms.squeeze(1)
        mel_spec = self.mel_transform(waveforms)
        log_mel = self.amplitude_to_db(mel_spec)
        return log_mel
