
from pathlib import Path

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio


if __name__ == "__main__": 
    model = load_silero_vad()

    # in_audio_path = "/data/project/FunDiagnosis/eggs/audio/download/raw_audio_copy/Parkinson/ID20_pd_3_0_1_readtext.wav"
    # wav = read_audio(in_audio_path, sampling_rate=16000)

    # speech_timestamps = get_speech_timestamps(
    #     wav, 
    #     model, 
    #     threshold=0.8, 
    #     min_speech_duration_ms=1000, 
    #     min_silence_duration_ms=1000, 
    #     speech_pad_ms=300, 
    #     return_seconds=False
    # )
    # for ts in speech_timestamps: 
    #     print(ts)

    data_path = Path("/data/project/FunDiagnosis/eggs/audio/download/speech-1/Parkinson-Patient-Speech-Dataset-master/denoised-speech-dataset")
    out_data_path = data_path.parent.parent.parent / "silero_sliced" / "Parkinson"
    out_data_path.mkdir(parents=True, exist_ok=True)
    for patient in data_path.iterdir(): 
        for file in patient.rglob("*.wav"):
            in_audio_path = file.as_posix()
            wav = read_audio(in_audio_path, sampling_rate=16000)
            speech_timestamps = get_speech_timestamps(
                wav, 
                model, 
                threshold=0.8, 
                min_speech_duration_ms=3000, 
                max_speech_duration_s=10, 
                min_silence_duration_ms=1000, 
                speech_pad_ms=100, 
                return_seconds=False
            )
            out_dir = out_data_path / patient.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, ts in enumerate(speech_timestamps): 
                start_sample = ts["start"]
                end_sample = ts["end"]
                out_audio_path = out_dir / f"{file.stem}_part{i}.wav"
                while out_audio_path.exists():
                    i += 1
                    out_audio_path = out_dir / f"{file.stem}_part{i}.wav"
                save_audio(out_audio_path, wav[start_sample:end_sample], sampling_rate=16000)
