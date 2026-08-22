import os
import argparse
import subprocess
import shutil
import numpy as np
import librosa
import soundfile as sf
import yt_dlp
import traceback


def download_track(query, output_filename="temp_song"):
    print(f"📥 Searching/downloading: '{query}'...")
    if not query.startswith("http"):
        query = f"ytsearch1:{query}"

    ydl_opts = {
        "format": "ba[ext=m4a]/bestaudio/best",
        "outtmpl": f"{output_filename}.%(ext)s",
        "remote_components": ["ejs:github"],
        "cookiefile": "cookies.txt",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])

    final_path = f"{output_filename}.wav"
    if os.path.exists(final_path):
        print("✅ Download complete.")
        return final_path
    else:
        raise Exception(
            "Error processing the audio file. Is ffmpeg installed?"
        )


def load_audio_stereo(path, target_sr=44100):
    y, sr = librosa.load(path, sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.vstack((y, y))
    return y.T, sr


def overlay_fx_stereo(base_audio, fx_audio, drop_sample, fx_volume=0.6, align="peak"):
    if align == "peak":
        fx_mono = np.mean(fx_audio, axis=1)
        climax_index = np.argmax(np.abs(fx_mono))
        start_sample = drop_sample - climax_index
    else:
        start_sample = drop_sample

    if start_sample < 0:
        fx_audio = fx_audio[abs(start_sample):, :]
        start_sample = 0

    end_sample = start_sample + fx_audio.shape[0]
    mixed_audio = np.copy(base_audio)

    if end_sample > mixed_audio.shape[0]:
        fx_audio = fx_audio[: mixed_audio.shape[0] - start_sample, :]
        end_sample = mixed_audio.shape[0]

    mixed_audio[start_sample:end_sample, :] += fx_audio * fx_volume
    mixed_audio = np.clip(mixed_audio, -1.0, 1.0)
    return mixed_audio


def sync_loop_bpm(loop_audio, loop_sr, song_audio, song_sr):
    print("🔄 Syncing loop BPM to the track...")
    song_mono = np.mean(song_audio, axis=1)
    loop_mono = np.mean(loop_audio, axis=1)

    song_tempo, _ = librosa.beat.beat_track(y=song_mono, sr=song_sr)
    loop_tempo, _ = librosa.beat.beat_track(y=loop_mono, sr=loop_sr)

    song_bpm = (
        float(song_tempo[0])
        if isinstance(song_tempo, np.ndarray)
        else float(song_tempo)
    )
    loop_bpm = (
        float(loop_tempo[0])
        if isinstance(loop_tempo, np.ndarray)
        else float(loop_tempo)
    )

    print(
        f"   Estimated BPM - Song: ~{song_bpm:.1f} | Loop: ~{loop_bpm:.1f}")

    if loop_bpm == 0:
        return loop_audio

    rate = song_bpm / loop_bpm
    stretched = librosa.effects.time_stretch(y=loop_audio.T, rate=rate)
    return stretched.T


def build_live_edit(original_path, intro_drums_path, sweeper_path, output_name):
    original_audio, sr = load_audio_stereo(original_path)

    try:
        intro_audio, _ = load_audio_stereo(intro_drums_path)
        intro_audio = sync_loop_bpm(intro_audio, sr, original_audio, sr)
    except Exception as e:
        print(
            f"⚠️ Could not load the loop ({e}). It will be skipped. Make sure you have {intro_drums_path}"
        )
        intro_audio = np.zeros((0, 2))

    extended_track = np.concatenate((intro_audio, original_audio), axis=0)
    drop_sample_index = intro_audio.shape[0]

    try:
        sweeper_audio, _ = load_audio_stereo(sweeper_path)
        final_audio = overlay_fx_stereo(
            base_audio=extended_track,
            fx_audio=sweeper_audio,
            drop_sample=drop_sample_index,
            fx_volume=0.75,
        )
    except Exception as e:
        print(f"⚠️ Could not load the sweeper ({e}). It will be skipped.")
        final_audio = extended_track

    sf.write(output_name, final_audio, sr)
    print(f"✅ Edit created successfully: {output_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Automatic DJ Extended Edit generator")
    parser.add_argument("query", nargs="+",
                        help="Song name or YouTube URL")
    parser.add_argument("--sweeper", default="assets/sweeper_fx.wav",
                        help="Path to your rising effect")
    args = parser.parse_args()

    song_query = " ".join(args.query)
    downloaded_path = "temp_song.wav"
    os.makedirs("assets", exist_ok=True)

    try:
        downloaded_path = download_track(song_query, "temp_song")
        output_name = f"EXTENDED_EDIT.wav"

        # Call the new function without passing an external loop
        build_live_edit_auto(
            original_path=downloaded_path,
            sweeper_path=args.sweeper,
            output_name=output_name
        )

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if os.path.exists(downloaded_path):
            os.remove(downloaded_path)


def extract_auto_loop(audio_path, target_sr=44100):
    print("🧠 Separating stems with AI (extracting drums)...")
    import sys

    # capture_output=True lets us see the real error if it fails
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "-d", "cpu", "-n",
            "htdemucs", "--two-stems=drums", audio_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        # If it fails, print the exact error raised by Demucs
        print(f"\n❌ AI ERROR DETAILS:\n{result.stderr}")
        raise Exception(
            "Stem separation failed (check the error above).")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    drums_path = os.path.join("separated", "htdemucs", base_name, "drums.wav")

    if not os.path.exists(drums_path):
        raise Exception(
            f"Drum file not found at path: {drums_path}")

    print("🥁 Analyzing energy and isolating the best bar...")
    drums_audio, sr = load_audio_stereo(drums_path, target_sr)
    drums_mono = np.mean(drums_audio, axis=1)

    tempo, beats = librosa.beat.beat_track(y=drums_mono, sr=sr)

    if len(beats) < 8:
        raise Exception(
            "No consistent rhythm was detected to create the loop.")

    max_energy = 0
    best_start_idx = 0

    for i in range(len(beats) - 4):
        s_start = librosa.frames_to_samples(beats[i])
        s_end = librosa.frames_to_samples(beats[i+4])

        segment = drums_mono[s_start:s_end]
        energy = np.sum(segment**2)

        if energy > max_energy:
            max_energy = energy
            best_start_idx = i

    loop_start = librosa.frames_to_samples(beats[best_start_idx])
    loop_end = librosa.frames_to_samples(beats[best_start_idx + 4])
    best_loop = drums_audio[loop_start:loop_end]

    intro_loop = np.tile(best_loop, (8, 1))

    shutil.rmtree("separated")

    return intro_loop


def build_live_edit_auto(original_path, sweeper_path, output_name):
    original_audio, sr = load_audio_stereo(original_path)

    try:
        # Generate the loop automatically from the same song
        intro_audio = extract_auto_loop(original_path, int(sr))
    except Exception as e:
        print(
            f"⚠️ Error generating automatic loop ({e}). The file will be created without an intro.")
        excepcion = traceback.format_exc()
        print(excepcion)
        intro_audio = np.zeros((0, 2))

    extended_track = np.concatenate((intro_audio, original_audio), axis=0)
    drop_sample_index = intro_audio.shape[0]

    try:
        sweeper_audio, _ = load_audio_stereo(sweeper_path)
        final_audio = overlay_fx_stereo(
            base_audio=extended_track,
            fx_audio=sweeper_audio,
            drop_sample=drop_sample_index,
            fx_volume=0.75
        )
    except Exception as e:
        print(f"⚠️ Could not load the sweeper ({e}). It will be skipped.")
        final_audio = extended_track

    sf.write(output_name, final_audio, sr)
    print(f"✅ Edit created successfully: {output_name}")


if __name__ == "__main__":
    main()
