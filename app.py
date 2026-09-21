import os
import threading
import queue
from flask import Flask, request, render_template_string
from dj_extended import download_track, build_live_edit_auto

app = Flask(__name__)
job_queue = queue.Queue()

# Worker que procesa 1 canción a la vez en segundo plano para cuidar la GPU


def worker():
    while True:
        song_query = job_queue.get()
        print(f"\n🎧 [COLA] Iniciando procesamiento: {song_query}")
        temp_path = "temp_song.wav"
        try:
            downloaded = download_track(song_query, "temp_song")
            # Guarda el archivo con un nombre limpio o único
            safe_name = "".join(
                [c if c.isalnum() else "_" for c in song_query])[:20]
            output_name = f"EDIT_{safe_name}.wav"

            build_live_edit_auto(
                original_path=downloaded,
                sweeper_path="assets/riser.mp3",
                output_name=output_name
            )
            print(f"🎉 [COLA] Listo: {output_name}")
        except Exception as e:
            print(f"❌ [COLA] Error con '{song_query}': {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            job_queue.task_done()


# Iniciar el hilo del worker al arrancar el servidor
threading.Thread(target=worker, daemon=True).start()

# HTML simple y responsive para celulares
HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pide tu tema al DJ</title>
    <style>
        body { font-family: sans-serif; background: #111; color: #fff; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px; box-sizing: border-box; }
        .card { background: #222; padding: 25px; border-radius: 12px; width: 100%; max-width: 400px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        input { width: 100%; padding: 14px; margin: 15px 0; border-radius: 8px; border: none; font-size: 16px; box-sizing: border-box; }
        button { width: 100%; padding: 14px; background: #e11d48; color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: bold; cursor: pointer; }
        .info { color: #888; font-size: 14px; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🎧 Pedir Canción</h2>
        <form method="POST" action="/pedir">
            <input type="text" name="song" placeholder="Ej: Feid - Luna o link" required autofocus>
            <button type="submit">Enviar al DJ</button>
        </form>
        <p class="info">Tu tema entrará a la cola y se procesará automáticamente con intro extendida.</p>
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/pedir", methods=["POST"])
def pedir():
    song = request.form.get("song", "").strip()
    if song:
        job_queue.put(song)
        posicion = job_queue.qsize()
        return f"""
        <body style="background:#111;color:#fff;font-family:sans-serif;text-align:center;padding:50px 20px;">
            <h2>✅ ¡Petición recibida!</h2>
            <p>Canción: <strong>{song}</strong></p>
            <p>Posición aproximada en cola: #{posicion}</p>
            <a href="/" style="color:#e11d48;text-decoration:none;font-weight:bold;">Pedir otra</a>
        </body>
        """
    return "Error: Canción vacía", 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
