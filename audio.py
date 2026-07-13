import os
import numpy as np
import soundfile as sf
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pydub import AudioSegment

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.res_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        residual = self.res_connection(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 2, 1).contiguous(), loss, encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.res3 = ResidualBlock(hidden_channels, hidden_channels)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.res1 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv1 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.res2 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.res3 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv3 = nn.ConvTranspose1d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = F.relu(self.conv1(x))
        x = self.res2(x)
        x = F.relu(self.conv2(x))
        x = self.res3(x)
        x = self.conv3(x)
        return x

class VQVAE2(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, num_embeddings=512, embedding_dim=64):
        super(VQVAE2, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.pre_vq_conv = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, in_channels)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights to produce smoother outputs without training"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        quantized, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, quantized

class ImprovedAudioSteganography:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 10
        self.bit_duration = 0.005  # 5ms per bit

        self.freq_pairs = [
            (17000, 17500),  # Pair 1
            (17750, 18250),  # Pair 2
            (18500, 19000),  # Pair 3
            (19250, 19750)  # Pair 4
        ]

        self.vqvae = VQVAE2()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vqvae.to(self.device)
        self.vqvae.eval()

        self.instrument_presets = {
            "piano": {"mix": [("sine", 0.5), ("triangle", 0.25), ("saw", 0.25)], "harmonics": [1, 2, 3],
                      "attack": 0.02, "decay": 0.25, "sustain": 0.35, "release": 0.12, "vibrato": (4.5, 0.25),
                      "noise": 0.0015},
            "organ": {"mix": [("sine", 0.7), ("square", 0.2), ("triangle", 0.1)], "harmonics": [1, 2, 4],
                      "attack": 0.05, "decay": 0.12, "sustain": 0.8, "release": 0.22, "vibrato": (5.0, 0.18),
                      "noise": 0.0004},
            "bell": {"mix": [("sine", 0.8), ("sine", 0.2)], "partials": [1, 2.7, 4.1, 5.3],
                     "attack": 0.01, "decay": 0.35, "sustain": 0.0, "release": 0.4, "vibrato": (6.5, 0.2),
                     "noise": 0.001},
            "pluck": {"mix": [("saw", 0.45), ("triangle", 0.35), ("sine", 0.2)], "harmonics": [1, 2, 3, 4],
                      "attack": 0.008, "decay": 0.2, "sustain": 0.2, "release": 0.12, "vibrato": (4.0, 0.12),
                      "noise": 0.0012},
            "pad": {"mix": [("triangle", 0.55), ("sine", 0.45)], "harmonics": [1, 2, 3],
                    "attack": 0.18, "decay": 0.25, "sustain": 0.75, "release": 0.3, "vibrato": (0.9, 0.15),
                    "noise": 0.0002},
            "synth": {"mix": [("saw", 0.55), ("square", 0.25), ("triangle", 0.2)], "harmonics": [1, 2, 3, 5],
                      "attack": 0.03, "decay": 0.18, "sustain": 0.55, "release": 0.2, "vibrato": (6.0, 0.22),
                      "noise": 0.0008},
            "sub": {"mix": [("sine", 0.85), ("triangle", 0.15)], "harmonics": [1, 2],
                    "attack": 0.02, "decay": 0.12, "sustain": 0.65, "release": 0.15, "vibrato": (0.5, 0.1),
                    "noise": 0.0},
        }

        self.end_marker = "<<END>>"
        self.last_seed = None

    def text_to_binary(self, text):
        return ''.join(format(ord(char), '08b') for char in text)

    def binary_to_text(self, binary):
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i + 8]
            if len(byte) == 8:
                text += chr(int(byte, 2))
        return text

    def generate_carrier(self, frequency, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        return 0.05 * np.sin(2 * np.pi * frequency * t)

    def _adsr_envelope(self, length, attack, decay, sustain, release):
        attack_len = max(1, int(length * attack))
        decay_len = max(1, int(length * decay))
        release_len = max(1, int(length * release))
        sustain_len = max(1, length - attack_len - decay_len - release_len)

        attack_curve = np.linspace(0, 1, attack_len, endpoint=False)
        decay_curve = np.linspace(1, sustain, decay_len, endpoint=False)
        sustain_curve = np.full(sustain_len, sustain)
        release_curve = np.linspace(sustain, 0, release_len)

        envelope = np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])
        if len(envelope) < length:
            envelope = np.pad(envelope, (0, length - len(envelope)), mode='edge')
        return envelope[:length]

    def _oscillator(self, freq, t, wave, phase_offset=0.0):
        phase = 2 * np.pi * freq * t + phase_offset
        if wave == "triangle":
            return 2 / np.pi * np.arcsin(np.sin(phase))
        if wave == "saw":
            return 2 * (t * freq - np.floor(0.5 + t * freq))
        if wave == "square":
            return np.sign(np.sin(phase))
        return np.sin(phase)

    def _synthesize_tone(self, freq, t, preset, accent, rng):
        rate, depth = preset.get("vibrato", (0.0, 0.0))
        phase_offset = depth * np.sin(2 * np.pi * rate * t)
        partials = preset.get("partials") or preset.get("harmonics", [1, 2, 3])

        tone = np.zeros_like(t)
        for wave, weight in preset.get("mix", [("sine", 1.0)]):
            partial_sum = np.zeros_like(t)
            for idx, multiplier in enumerate(partials):
                attenuation = 1.0 / ((idx + 1) ** 1.2)
                partial_sum += attenuation * self._oscillator(freq * multiplier, t, wave, phase_offset)
            tone += weight * partial_sum

        envelope = self._adsr_envelope(
            len(t),
            preset.get("attack", 0.02),
            preset.get("decay", 0.2),
            preset.get("sustain", 0.5),
            preset.get("release", 0.1),
        )

        tone = tone * envelope * accent
        noise_amount = preset.get("noise", 0.0)
        if noise_amount:
            tone += rng.normal(0, noise_amount, len(t))
        return tone

    def _synthesize_kick(self, t, accent, rng):
        freqs = np.linspace(150, 40, len(t))
        phase = np.cumsum(freqs * 2 * np.pi / self.sample_rate)
        kick = np.sin(phase)
        env = self._adsr_envelope(len(t), 0.005, 0.1, 0.0, 0.1)
        return kick * env * accent * 1.5

    def _synthesize_snare(self, t, accent, rng):
        noise = rng.normal(0, 1, len(t))
        sos_hp = signal.butter(4, 1000, 'highpass', fs=self.sample_rate, output='sos')
        noise = signal.sosfilt(sos_hp, noise)
        tone = np.sin(2 * np.pi * 200 * t)
        env = self._adsr_envelope(len(t), 0.005, 0.15, 0.0, 0.1)
        return (noise * 0.8 + tone * 0.2) * env * accent * 1.2

    def _synthesize_hihat(self, t, accent, rng):
        noise = rng.normal(0, 1, len(t))
        sos_hp = signal.butter(4, 7000, 'highpass', fs=self.sample_rate, output='sos')
        noise = signal.sosfilt(sos_hp, noise)
        env = self._adsr_envelope(len(t), 0.005, 0.05, 0.0, 0.05)
        return noise * env * accent * 0.3

    def generate_improved_melody(self, instrument="pad", timbre_mode="single", genre="pop", seed=None):
        """Generate a higher quality instrumental melody with selectable timbres and genres."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        melody = np.zeros_like(t)

        note_frequencies = {
            "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13, "E": 329.63,
            "F": 349.23, "F#": 369.99, "G": 392.00, "G#": 415.30, "A": 440.00,
            "A#": 466.16, "B": 493.88
        }
        note_names = list(note_frequencies.keys())

        chord_structures = {
            "major": [0, 4, 7],
            "minor": [0, 3, 7],
            "major7": [0, 4, 7, 11],
            "minor7": [0, 3, 7, 10],
            "dom7": [0, 4, 7, 10]
        }

        genres = {
            "pop": {
                "progressions": [
                    [(0, "major"), (7, "major"), (9, "minor"), (5, "major")],
                    [(0, "major"), (5, "major"), (7, "major"), (0, "major")]
                ],
                "drums": [[1,0,0,0, 0,0,0,0], [0,0,0,0, 1,0,0,0], [1,0,1,0, 1,0,1,0]],
                "bass_patterns": [[0, 0, 7, 0, 5, 7, 0, 0]],
            },
            "jazz": {
                "progressions": [
                    [(2, "minor7"), (7, "dom7"), (0, "major7"), (0, "major7")],
                ],
                "drums": [[1,0,0,0, 0,0,0,0], [0,0,0,0, 1,0,0,0], [1,0,1,1, 1,0,1,1]],
                "bass_patterns": [[0, 5, 7, 5, 0, 7, 5, 0]],
            },
            "lofi": {
                "progressions": [
                    [(0, "major7"), (4, "minor7"), (5, "major7"), (5, "major7")],
                    [(2, "minor7"), (5, "major7"), (0, "major7"), (9, "minor7")]
                ],
                "drums": [[1,0,0,0, 0,0,1,0], [0,0,0,0, 1,0,0,0], [1,0,1,0, 1,0,1,0]],
                "bass_patterns": [[0, 0, 12, 0, 7, 5, 0, -5]],
            },
            "classical": {
                "progressions": [
                    [(0, "major"), (5, "major"), (7, "dom7"), (0, "major")],
                ],
                "drums": [[0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0]],
                "bass_patterns": [[0, 7, 0, 7, 0, 7, 0, 7]],
            }
        }

        genre_data = genres.get(genre.lower(), genres["pop"])
        progressions = genre_data["progressions"]
        bass_patterns = genre_data["bass_patterns"]
        drums = genre_data["drums"]

        base_index = int(rng.integers(0, len(note_names)))
        progression = progressions[rng.integers(0, len(progressions))]
        chord_progression = progression * (8 // len(progression))
        if len(chord_progression) < 8:
            chord_progression += progression[: 8 - len(chord_progression)]

        bass_pattern = bass_patterns[rng.integers(0, len(bass_patterns))]

        available_instruments = [key for key in self.instrument_presets.keys() if key != "sub"]
        chord_duration = self.duration / len(chord_progression)
        beat_duration = chord_duration / 8

        for i, (interval, chord_type) in enumerate(chord_progression):
            root_index = (base_index + interval) % len(note_names)
            root_name = note_names[root_index]
            root_freq = note_frequencies[root_name]

            if timbre_mode == "cycle":
                chord_instrument = available_instruments[i % len(available_instruments)]
            elif timbre_mode == "random":
                chord_instrument = available_instruments[rng.integers(0, len(available_instruments))]
            else:
                chord_instrument = instrument

            chord_preset = self.instrument_presets.get(chord_instrument, self.instrument_presets["pad"])
            bass_preset = self.instrument_presets["sub"]

            for beat in range(8):
                start_time = i * chord_duration + beat * beat_duration
                start_idx = int(start_time * self.sample_rate)
                beat_length = int(beat_duration * self.sample_rate)
                end_idx = min(start_idx + beat_length, len(melody))
                if end_idx <= start_idx:
                    continue

                beat_t = np.linspace(0, beat_duration, end_idx - start_idx, False)
                beat_signal = np.zeros_like(beat_t)

                # Chords (pad/keys)
                if beat % 2 == 0:
                    for semitone_offset in chord_structures[chord_type]:
                        note_freq = root_freq * (2 ** (semitone_offset / 12))
                        beat_signal += self._synthesize_tone(note_freq, beat_t, chord_preset, 0.6, rng)

                # Bass
                bass_note = root_freq * 0.5 * (2 ** (bass_pattern[beat] / 12))
                beat_signal += self._synthesize_tone(bass_note, beat_t, bass_preset, 0.8, rng)

                # Drums
                if drums[0][beat]:
                    beat_signal += self._synthesize_kick(beat_t, 1.0, rng)
                if drums[1][beat]:
                    beat_signal += self._synthesize_snare(beat_t, 1.0, rng)
                if drums[2][beat]:
                    beat_signal += self._synthesize_hihat(beat_t, 1.0, rng)

                melody[start_idx:end_idx] += beat_signal[: end_idx - start_idx]

        reverb_length = int(0.32 * self.sample_rate)
        reverb_envelope = np.exp(-np.linspace(0, 6, reverb_length))
        reverb_envelope = reverb_envelope / np.sum(reverb_envelope)
        melody_with_reverb = np.convolve(melody, reverb_envelope, mode='full')[:len(melody)]

        # Vinyl crackle
        subtle_noise = rng.normal(0, 0.001, len(melody_with_reverb))
        sos_noise = signal.butter(4, [1000, 8000], 'bandpass', fs=self.sample_rate, output='sos')
        filtered_noise = signal.sosfilt(sos_noise, subtle_noise)
        melody_with_reverb += filtered_noise * 0.5

        sos_smooth = signal.butter(3, 12000, 'lowpass', fs=self.sample_rate, output='sos')
        melody_with_reverb = signal.sosfilt(sos_smooth, melody_with_reverb)

        threshold = 0.55
        ratio = 1.6
        attack = 0.01
        release = 0.1
        attack_coef = np.exp(-1.0 / (self.sample_rate * attack))
        release_coef = np.exp(-1.0 / (self.sample_rate * release))
        env = 0
        compressed = np.zeros_like(melody_with_reverb)

        for i, sample in enumerate(melody_with_reverb):
            env_in = abs(sample)
            if env_in > env:
                env = attack_coef * env + (1 - attack_coef) * env_in
            else:
                env = release_coef * env + (1 - release_coef) * env_in

            if env <= threshold:
                gain = 1.0
            else:
                gain = 1.0 + (1.0 / ratio - 1.0) * (env - threshold) / env
            compressed[i] = sample * gain

        melody_with_reverb = compressed

        sos_mid = signal.butter(2, [300, 3000], 'bandpass', fs=self.sample_rate, output='sos')
        melody_mid = signal.sosfilt(sos_mid, melody_with_reverb)
        melody_with_reverb = 0.85 * melody_with_reverb + 0.15 * melody_mid
        melody_with_reverb = melody_with_reverb / np.max(np.abs(melody_with_reverb)) * 0.8

        return melody_with_reverb

    def embed_message(self, message, output_file="output.wav", instrument="pad", timbre_mode="single", genre="pop", seed=None):
        if seed is None:
            seed = int(time.time() * 1000) % (2 ** 32)
        self.last_seed = seed

        message_with_marker = message + self.end_marker
        binary_message = self.text_to_binary(message_with_marker)

        bits_per_channel = int(self.duration / self.bit_duration)
        max_bits = bits_per_channel * len(self.freq_pairs)
        max_chars = max_bits // 8

        if len(binary_message) > max_bits:
            raise ValueError(f"Message too long. Maximum length is {max_chars - len(self.end_marker)} characters")

        melody = self.generate_improved_melody(
            instrument=instrument,
            timbre_mode=timbre_mode,
            genre=genre,
            seed=seed
        )

        def calculate_masking_threshold(audio_segment, freq):
            fft = np.abs(np.fft.rfft(audio_segment))
            freqs = np.fft.rfftfreq(len(audio_segment), 1 / self.sample_rate)
            idx = np.argmin(np.abs(freqs - freq))
            window_size = 10
            start_idx = max(0, idx - window_size)
            end_idx = min(len(fft), idx + window_size)
            nearby_energy = np.mean(fft[start_idx:end_idx])
            return min(0.3, max(0.05, nearby_energy / np.max(fft) * 0.2))

        data_signal = np.zeros(len(melody))
        samples_per_bit = int(self.bit_duration * self.sample_rate)
        padding_length = (-len(binary_message)) % 4
        binary_message += '0' * padding_length

        for i in range(0, len(binary_message), 4):
            bit_group = binary_message[i:i + 4]
            start = (i // 4) * samples_per_bit
            end = start + samples_per_bit

            if start >= len(melody):
                break

            for j, bit in enumerate(bit_group):
                freq = self.freq_pairs[j][1] if bit == '1' else self.freq_pairs[j][0]
                segment = melody[start:end] if end <= len(melody) else melody[start:]
                mask_level = calculate_masking_threshold(segment, freq)

                carrier = mask_level * np.sin(2 * np.pi * freq * np.linspace(0, self.bit_duration, samples_per_bit, False))
                fade_samples = min(int(0.0005 * self.sample_rate), len(carrier) // 10)
                carrier[:fade_samples] *= np.linspace(0, 1, fade_samples)
                carrier[-fade_samples:] *= np.linspace(1, 0, fade_samples)

                if end <= len(data_signal):
                    data_signal[start:end] += carrier
                else:
                    data_signal[start:] += carrier[:len(data_signal) - start]

        sos = signal.butter(6, [40, 15000], 'bandpass', fs=self.sample_rate, output='sos')
        filtered_melody = signal.sosfilt(sos, melody)
        filtered_melody = filtered_melody / np.max(np.abs(filtered_melody)) * 0.82

        sos_aa = signal.butter(6, [16500, 20000], 'bandpass', fs=self.sample_rate, output='sos')
        filtered_data = signal.sosfilt(sos_aa, data_signal)

        audio = filtered_melody + filtered_data

        threshold = 0.92
        ratio = 3.0

        sos_low = signal.butter(3, 500, 'lowpass', fs=self.sample_rate, output='sos')
        sos_mid = signal.butter(3, [500, 8000], 'bandpass', fs=self.sample_rate, output='sos')
        sos_high = signal.butter(3, 8000, 'highpass', fs=self.sample_rate, output='sos')

        audio_low = signal.sosfilt(sos_low, audio)
        audio_mid = signal.sosfilt(sos_mid, audio)
        audio_high = signal.sosfilt(sos_high, audio)

        for band in [audio_low, audio_mid, audio_high]:
            mask = np.abs(band) > threshold
            if np.any(mask):
                band[mask] = np.sign(band[mask]) * (threshold + (np.abs(band[mask]) - threshold) / ratio)

        audio = audio_low + audio_mid + audio_high
        audio = audio / np.max(np.abs(audio)) * 0.95

        dither_amplitude = 1.0 / (2 ** 15)
        dither = np.random.uniform(-dither_amplitude, dither_amplitude, len(audio))
        audio += dither

        if output_file.lower().endswith(".mp3"):
            temp_wav = output_file + ".temp.wav"
            sf.write(temp_wav, audio, self.sample_rate, subtype='PCM_24')
            audio_segment = AudioSegment.from_wav(temp_wav)
            audio_segment.export(output_file, format="mp3", bitrate="320k")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        else:
            sf.write(output_file, audio, self.sample_rate, subtype='PCM_24')
        return audio

    def extract_message(self, audio_file, max_bits=None):
        if audio_file.lower().endswith(".mp3"):
            temp_wav = audio_file + ".temp.wav"
            audio_segment = AudioSegment.from_mp3(audio_file)
            audio_segment.export(temp_wav, format="wav")
            audio, _ = sf.read(temp_wav)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        else:
            audio, _ = sf.read(audio_file)

        sos = signal.butter(8, 16000, 'highpass', fs=self.sample_rate, output='sos')
        filtered_audio = signal.sosfilt(sos, audio)

        samples_per_bit = int(self.bit_duration * self.sample_rate)
        binary_message = ''

        if max_bits is None:
            max_bit_groups = len(filtered_audio) // samples_per_bit
            max_bits = max_bit_groups * 4

        for i in range((max_bits + 3) // 4):
            start = i * samples_per_bit
            end = start + samples_per_bit

            if end > len(filtered_audio):
                break

            segment = filtered_audio[start:end]
            segment = segment * np.hanning(len(segment))

            n_fft = 2048
            freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)
            fft = np.abs(np.fft.rfft(segment, n=n_fft))

            for j, (freq_0, freq_1) in enumerate(self.freq_pairs):
                idx_0 = np.argmin(np.abs(freqs - freq_0))
                idx_1 = np.argmin(np.abs(freqs - freq_1))
                window = 2
                energy_0 = np.sum(fft[max(0, idx_0 - window):idx_0 + window + 1])
                energy_1 = np.sum(fft[max(0, idx_1 - window):idx_1 + window + 1])

                binary_message += '1' if energy_1 > energy_0 else '0'
                if len(binary_message) >= max_bits:
                    break

            if len(binary_message) >= max_bits:
                break

        full_text = self.binary_to_text(binary_message)

        if self.end_marker in full_text:
            extracted_message = full_text.split(self.end_marker)[0]
            return extracted_message
        else:
            print("Warning: End marker not found. Returning all extracted data.")
            return full_text

def test_improved_steganography(message):
    stego = ImprovedAudioSteganography()
    try:
        print("Generating audio with hidden message using improved melody generation...")
        audio = stego.embed_message(message, "hidden_message_improved.wav")
        print("\nExtracting hidden message without knowing its length...")
        extracted_message = stego.extract_message("hidden_message_improved.wav")
        print(f"\nOriginal message: {message}")
        print(f"Extracted message: {extracted_message}")
        print(f"\nMessage length: {len(message)} characters")
        print(f"Success: {message == extracted_message}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_message = "hello everyone!."
    test_improved_steganography(test_message)
