import os
import sys
import shutil
import argparse
import subprocess
import numpy as np
import librosa
import soundfile as sf
import yt_dlp


def download_track(query, output_filename="temp_song"):
    print(f"📥 Buscando/Descargando: '{query}'...")
    if not query.startswith("http"):
        query = f"ytsearch1:{query}"

    ydl_opts = {
        'format': 'ba[ext=m4a]/bestaudio/best',
        'outtmpl': f'{output_filename}.%(ext)s',
        'remote_components': ['ejs:github'],
        'extractor_args': {'youtube': {'player_client': ['tv', 'ios']}},
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': False,
        'noplaylist': True
    }

    # Si tienes un archivo cookies.txt en la raíz, lo aprovechará
    if os.path.exists("cookies.txt"):
        ydl_opts['cookiefile'] = 'cookies.txt'

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])

    final_path = f"{output_filename}.wav"
    if os.path.exists(final_path):
        print("✅ Descarga completada.")
        return final_path
    else:
        raise Exception(
            "Error al procesar el archivo de audio. ¿Está instalado ffmpeg?")


def load_audio_stereo(path, target_sr=44100):
    y, sr = librosa.load(path, sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.vstack((y, y))
    return y.T, sr


def apply_micro_fades(audio, fade_ms=5, sr=44100):
    fade_samples = int((fade_ms / 1000.0) * sr)
    if len(audio) < fade_samples * 2:
        return audio

    fade_in = np.linspace(0.0, 1.0, fade_samples)[:, np.newaxis]
    fade_out = np.linspace(1.0, 0.0, fade_samples)[:, np.newaxis]

    audio_faded = np.copy(audio)
    audio_faded[:fade_samples] *= fade_in
    audio_faded[-fade_samples:] *= fade_out
    return audio_faded


def overlay_fx_stereo(base_audio, fx_audio, drop_sample, fx_volume=0.75, align='peak'):
    if align == 'peak':
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
        fx_audio = fx_audio[:mixed_audio.shape[0] - start_sample, :]
        end_sample = mixed_audio.shape[0]

    mixed_audio[start_sample:end_sample, :] += (fx_audio * fx_volume)
    mixed_audio = np.clip(mixed_audio, -1.0, 1.0)
    return mixed_audio


def extract_auto_loop(audio_path, target_sr=44100):
    print("🧠 Separando stems con IA (Extrayendo batería vía GPU)...")

    result = subprocess.run(
        [sys.executable, "-m", "demucs", "-n",
            "htdemucs", "--two-stems=drums", audio_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"\n❌ DETALLE DEL ERROR DE IA:\n{result.stderr}")
        raise Exception("Fallo en la separación de stems.")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    drums_path = os.path.join("separated", "htdemucs", base_name, "drums.wav")

    if not os.path.exists(drums_path):
        raise Exception(
            f"No se encontró el archivo de batería en: {drums_path}")

    print("🥁 Buscando 4 compases de beat continuo y uniforme...")
    drums_audio, sr = load_audio_stereo(drums_path, target_sr)
    drums_mono = np.mean(drums_audio, axis=1)

    # 1. Detección con BPM rígido
    tempo, beats = librosa.beat.beat_track(
        y=drums_mono, sr=sr, start_bpm=100.0, tightness=100)
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    print(f"   BPM Detectado: ~{bpm:.1f}")

    # 2. 4 compases = 16 beats en compás 4/4
    samples_per_beat = (60.0 / bpm) * sr
    four_bars_samples = int(round(samples_per_beat * 16))
    one_bar_samples = int(round(samples_per_beat * 4))

    if len(beats) < 20:
        raise Exception(
            "La pista no tiene suficientes beats detectados para extraer 4 compases.")

    # 3. Evaluar bloques de 16 beats buscando consistencia (evitar buildups/redobles)
    best_score = -1
    best_start_sample = 0

    # Descartamos los primeros 2 compases para evitar intros sin base
    start_search_idx = min(8, len(beats) // 4)

    for idx in range(start_search_idx, len(beats) - 16):
        s_start = librosa.frames_to_samples(beats[idx])
        s_end = s_start + four_bars_samples

        if s_end > len(drums_mono):
            break

        # Dividimos los 4 compases en sus 4 compases individuales para medir estabilidad
        bar_energies = []
        for b in range(4):
            b_start = s_start + (b * one_bar_samples)
            b_end = b_start + one_bar_samples
            segment = drums_mono[b_start:b_end]
            bar_energies.append(np.mean(segment**2))

        avg_energy = np.mean(bar_energies)
        # Si sube mucho (buildup crescendo), std es alta
        std_energy = np.std(bar_energies)

        # Umbral mínimo para descartar silencios o partes sin batería
        if avg_energy < 1e-4:
            continue

        # Puntaje: premia energía sólida y penaliza variaciones bruscas entre compases
        # (coeficiente de variación: desviación relativa baja = ritmo parejo)
        coef_var = std_energy / (avg_energy + 1e-6)

        # Queremos avg_energy alto, pero coef_var bajo (ritmo estable)
        score = avg_energy / (1.0 + coef_var * 5.0)

        if score > best_score:
            best_score = score
            best_start_sample = s_start

    if best_score == -1:
        # Fallback al primer golpe con energía si no hubo uno ideal
        best_start_sample = librosa.frames_to_samples(beats[start_search_idx])

    # 4. Extraer exactamente los 4 compases matemáticos
    four_bars_loop = drums_audio[best_start_sample:
                                 best_start_sample + four_bars_samples]
    four_bars_loop = apply_micro_fades(four_bars_loop, fade_ms=5, sr=sr)

    # 5. Repetir 2 veces para completar la intro extendida de 8 compases (32 beats)
    intro_loop = np.tile(four_bars_loop, (2, 1))

    if os.path.exists("separated"):
        shutil.rmtree("separated")

    return intro_loop


def build_live_edit_auto(original_path, sweeper_path, output_name):
    original_audio, sr = load_audio_stereo(original_path)

    try:
        intro_audio = extract_auto_loop(original_path, sr)
    except Exception as e:
        print(
            f"⚠️ Error generando loop automático ({e}). Se omitirá la intro.")
        intro_audio = np.zeros((0, 2))

    extended_track = np.concatenate((intro_audio, original_audio), axis=0)
    drop_sample_index = intro_audio.shape[0]

    if os.path.exists(sweeper_path):
        sweeper_audio, _ = load_audio_stereo(sweeper_path, sr)
        final_audio = overlay_fx_stereo(
            base_audio=extended_track,
            fx_audio=sweeper_audio,
            drop_sample=drop_sample_index,
            fx_volume=0.75
        )
    else:
        print(
            f"⚠️ No se encontró {sweeper_path}. Se omite el efecto de subida.")
        final_audio = extended_track

    sf.write(output_name, final_audio, sr)
    print(f"✅ Edit creado con éxito: {output_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generador rápido de DJ Extended Edits")
    parser.add_argument("query", nargs="+",
                        help="Nombre de la canción o URL de YouTube")
    parser.add_argument("--sweeper", default="assets/riser.mp3",
                        help="Ruta de tu efecto de subida")
    args = parser.parse_args()

    song_query = " ".join(args.query)
    downloaded_path = "temp_song.wav"
    os.makedirs("assets", exist_ok=True)

    try:
        downloaded_path = download_track(song_query, "temp_song")
        output_name = "EXTENDED_EDIT.wav"
        print("🎛️ Procesando audio y ensamblando Extended Edit...")

        build_live_edit_auto(
            original_path=downloaded_path,
            sweeper_path=args.sweeper,
            output_name=output_name
        )

    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")
    finally:
        if os.path.exists(downloaded_path):
            os.remove(downloaded_path)


if __name__ == "__main__":
    main()
