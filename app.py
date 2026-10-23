import os
import re
import tempfile
from pathlib import Path
from collections import Counter
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
ALLOWED = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
MODEL = None

PAGE = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#172554">
<title>Analizador multilingüe de audio</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#182235;font:16px system-ui,-apple-system,Segoe UI,sans-serif}
header{background:#172554;color:white;padding:26px 18px}header .wrap,main{max-width:760px;margin:auto}h1{font-size:1.55rem;margin:0 0 8px}header p{margin:0;color:#dbeafe;line-height:1.45}
main{padding:20px 14px 40px}.card{background:white;border:1px solid #dce3ee;border-radius:18px;padding:18px;margin-bottom:16px;box-shadow:0 4px 14px #13203a0a}
label{display:block;font-weight:650;margin:12px 0 7px}input,select,button{font:inherit;width:100%;padding:12px;border-radius:10px;border:1px solid #bdc8d8;background:white}
button{border:0;background:#1d4ed8;color:white;font-weight:700;margin-top:16px;cursor:pointer}button:disabled{opacity:.6}small,.muted{color:#5c677b}.notice{background:#fff7e6;border:1px solid #f3d49a;border-radius:12px;padding:12px;line-height:1.45}
#status{margin-top:12px;white-space:pre-wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fc;padding:12px;border-radius:10px}
h2{font-size:1.15rem;margin:0 0 12px}.item{padding:10px 0;border-bottom:1px solid #e6eaf1}.item:last-child{border:0}
</style>
</head>
<body>
<header><div class="wrap"><h1>Analizador multilingüe de audio</h1><p>Transcripción y observaciones acústicas para revisión humana.</p></div></header>
<main>
<section class="card">
<h2>Analizar un audio</h2>
<p class="muted">Selecciona un archivo de voz de hasta 25 MB. La primera transcripción puede tardar mientras se carga el modelo.</p>
<form id="form">
<label for="audio">Archivo de audio</label><input id="audio" name="audio" type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm,.mp4" required>
<label for="language">Idioma de la voz</label>
<select id="language" name="language"><option value="auto">Detectar automáticamente</option><option value="es">Español</option><option value="en">Inglés</option><option value="fr">Francés</option><option value="pt">Portugués</option><option value="de">Alemán</option><option value="it">Italiano</option><option value="ja">Japonés</option><option value="zh">Chino</option><option value="ru">Ruso</option><option value="ar">Árabe</option></select>
<button id="submit" type="submit">Analizar audio</button>
<div id="status" role="status"></div>
</form>
</section>
<section class="notice"><strong>Importante:</strong> esta herramienta no detecta ni demuestra mentiras. El tono y las pausas no son pruebas de engaño. Verifica la transcripción y cualquier dato con fuentes independientes. No subas audio sin permiso.</section>
<section class="card" id="results" hidden><h2>Resultados</h2><div id="output"></div><button id="download" type="button">Descargar informe JSON</button></section>
<footer class="muted" style="text-align:center;font-size:.85rem;padding:10px">Prototipo experimental. Evita subir información sensible.</footer>
</main>
<script>
const form=document.querySelector('#form'),statusEl=document.querySelector('#status'),submit=document.querySelector('#submit'),results=document.querySelector('#results'),output=document.querySelector('#output'),download=document.querySelector('#download');let report=null;
form.addEventListener('submit',async e=>{e.preventDefault();if(!document.querySelector('#audio').files.length)return;submit.disabled=true;statusEl.textContent='Analizando… la primera ejecución puede tardar varios minutos.';results.hidden=true;
try{const r=await fetch('/analyze',{method:'POST',body:new FormData(form)});const data=await r.json();if(!r.ok)throw new Error(data.error||'No se pudo analizar el audio.');report=data;output.replaceChildren();
const add=(title,value)=>{const div=document.createElement('div');div.className='item';const strong=document.createElement('strong');strong.textContent=title;const pre=document.createElement('pre');pre.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);div.append(strong,pre);output.append(div)};
add('Idioma detectado',data.idioma_detectado+' (estimación '+data.confianza_idioma_estimada+')');add('Transcripción',data.transcripcion);add('Características acústicas',data.caracteristicas_acusticas);add('Elementos para verificar',data.contenido_para_verificar);add('Limitaciones',data.limitaciones);results.hidden=false;statusEl.textContent='Análisis terminado.'}
catch(err){statusEl.textContent='Error: '+err.message}finally{submit.disabled=false}});
download.addEventListener('click',()=>{if(!report)return;const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='informe_audio.json';a.click();URL.revokeObjectURL(url)});
</script></body></html>
"""

def get_model():
    global MODEL
    if MODEL is None:
        from faster_whisper import WhisperModel
        MODEL = WhisperModel(os.environ.get("WHISPER_MODEL", "tiny"), device="cpu", compute_type="int8")
    return MODEL

def acoustic_features(path):
    import librosa
    import numpy as np
    audio, sr = librosa.load(path, sr=16000, mono=True)
    if audio.size == 0:
        raise ValueError("El archivo de audio está vacío.")
    rms = librosa.feature.rms(y=audio)[0]
    try:
        f0 = librosa.yin(audio, fmin=65, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 >= 65) & (f0 <= 400)]
    except Exception:
        f0 = np.array([])
    hop = 160
    frame_rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=hop)[0]
    threshold = max(float(np.percentile(frame_rms, 20)), 0.008)
    quiet = frame_rms < threshold
    changes = np.diff(np.r_[False, quiet, False].astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    pauses = [(b-a)*hop/sr for a,b in zip(starts, ends)]
    long_pauses = [p for p in pauses if p >= .25]
    return {
        "Duración (segundos)": round(len(audio)/sr, 2),
        "Tono mediano estimado (Hz)": round(float(np.median(f0)), 2) if f0.size else None,
        "Variación aproximada del tono (Hz)": round(float(np.std(f0)), 2) if f0.size else None,
        "Intensidad RMS media": round(float(np.mean(rms)), 5),
        "Pausas estimadas de 250 ms o más": len(long_pauses),
        "Duración total de esas pausas (segundos)": round(float(sum(long_pauses)), 2),
        "Nota": "Son estimaciones acústicas; no son indicadores fiables de mentira."
    }

def content_features(text):
    nums = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    dates = re.findall(r"\b(?:hoy|ayer|mañana|lunes|martes|miércoles|jueves|viernes|sábado|domingo|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}:\d{2})\b", text, flags=re.IGNORECASE)
    words = re.findall(r"\b[a-záéíóúüñ]{4,}\b", text.lower())
    repeated = {w:c for w,c in Counter(words).items() if c > 1}
    repeated = dict(sorted(repeated.items(), key=lambda x: -x[1])[:15])
    return {"Números mencionados para verificar": nums, "Referencias temporales para verificar": dates, "Palabras repetidas": repeated, "Nota": "Estos elementos requieren contexto y corroboración independiente."}

@app.get("/")
def home():
    return render_template_string(PAGE)

@app.post("/analyze")
def analyze():
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify(error="Selecciona un archivo de audio."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(error="Formato no compatible. Prueba WAV, MP3, M4A, OGG, FLAC o WEBM."), 400
    lang = request.form.get("language", "auto")
    lang = None if lang == "auto" else lang
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            f.save(tmp.name)
            temp_path = tmp.name
        acoustic = acoustic_features(temp_path)
        model = get_model()
        segments, info = model.transcribe(temp_path, language=lang, beam_size=5, vad_filter=True)
        segs = [{"inicio_s": round(float(s.start), 2), "fin_s": round(float(s.end), 2), "texto": s.text.strip()} for s in segments]
        transcript = " ".join(s["texto"] for s in segs)
        return jsonify({"archivo": f.filename, "idioma_detectado": info.language,
            "confianza_idioma_estimada": round(float(info.language_probability), 3),
            "transcripcion": transcript, "segmentos": segs,
            "caracteristicas_acusticas": acoustic, "contenido_para_verificar": content_features(transcript),
            "limitaciones": ["Esta herramienta no detecta ni demuestra mentiras.",
            "La transcripción automática puede contener errores.",
            "Las características acústicas cambian según la persona, el idioma, la emoción y la grabación.",
            "Los elementos del contenido requieren corroboración independiente."]})
    except Exception:
        app.logger.exception("Fallo en análisis")
        return jsonify(error="No se pudo analizar el archivo. Prueba con un audio más corto o de otro formato."), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.errorhandler(413)
def too_large(_):
    return jsonify(error="El archivo supera el límite de 25 MB."), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
