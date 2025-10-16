import os
import csv
import math
import warnings
from typing import List, Tuple, Optional
import numpy as np
import librosa
from textgrid import TextGrid
from functools import lru_cache
import torch
import torchaudio

# =========== #
# Directorios #
# =========== #

# Se definen los nombres de las carpetas de entrada (INPUT_ROOT) y salida (OUTPUT_ROOT).

INPUT_ROOT = "DTW Entrada"
OUTPUT_ROOT = "DTW Salida"

# ======================================================= #
# Parámetros para el algoritmo Dynamic Time Warping (DTW) #
# ======================================================= #

# Se definen los parámetros que utilizara librosa para el algoritmo Dynamic Time Warping (DTW):

# SR: Frecuencia de muestreo (en hercios (Hz)).
# WIN_SEC: Tamaño de ventana (en segundos (s)).
# HOP_SEC: Desplazamiento de ventana (en segundos (s)). La cantidad de muestras por segundo para analizar.
# N_FFT: Número de muestras por ventana de la Transformada Rápida de Fourier (FFT). Controla la resolución espectral
# de la transformada de Fourier de tiempo corto (STFT) dentro del espectrograma Mel.
# WIN_LENGHT: Número de muestras de la función de ventana.
# HOP_LENGTH: Número de muestras de desplazamiento entre ventanas.
# N_MELS: Número de bandas de frecuencia Mel para el espectrograma Mel.
# N_MFCC: Número de Coeficientes Cepstrales de Frecuencia Mel (MFCC) extraídos del espectrograma Mel.
# WINDOW: La función de ventana antes de aplicar el FFT.
# FMIN, FMAX: El rango de frecuencia para el banco de filtros de Mel (en hercios (Hz)).
# DTW_METRIC: La distancia métrica para el alineamiento del DTW.
# MIN_FRAMES: La cantidad de muestras mínimas por segmento (extendiendo muestras que no alcancen este valor).

SR = 16000
WIN_SEC = 0.025  
HOP_SEC = 0.010  
N_FFT = int(round(SR * WIN_SEC)) 
WIN_LENGTH = N_FFT
HOP_LENGTH = int(round(SR * HOP_SEC))
N_MELS = 24
N_MFCC = 12
WINDOW = "hamming"
FMIN, FMAX = 300, 3400
DTW_METRIC = "euclidean"
MIN_FRAMES = 3

# =========================================================== #
# Definiciones de lectura y organizacion de archivos TextGrid #
# =========================================================== #

# === Definición de buscador de nivel (tier) de fonemas en archivos TextGrid === #

# Los archivos TextGrid pueden tener múltiples niveles (tiers) o estar configurados de formas distintas. Esta definición ayuda a
# a buscar nivel (tier) donde específicamente se encuentran los fonemas y sus intervalos temporales.

# Inicialmente, busca un tier cuyo nombre coincida con nombres comunes para fonemas:
# {"phones", "phoneme", "phonemes", "phone", "segment", "segments"}.
# Si encuentra coincidencia exacta (ignorando mayúsculas/minúsculas y espacios), devuelve ese índice de inmediato.

# Si no se encuentra ninguna coincidencia por nombre, entonces recorre todos los tiers de tipo IntervalTier y cuenta cuántos
# intervalos tiene cada uno. Cuando encuentre el índice del tier con mayor cantidad de intervalos, devuelve ese índice.
# Esto se hace bajo la suposición de que el tier fonémico suele tener la mayor segmentación (es decir, más intervalos).

def textgrid_tier_phoneme_finder(tg: TextGrid) -> Optional[int]:
    candidates = {"phones", "phoneme", "phonemes", "phone", "segment", "segments"}
    for i, tier in enumerate(tg.tiers):
        if hasattr(tier, "name") and str(tier.name).strip().lower() in candidates:
            return i
    best_idx, best_n = None, -1
    for i, tier in enumerate(tg.tiers):
        if tier.__class__.__name__.lower().endswith("intervaltier"):
            n = len(getattr(tier, "intervals", []))
            if n > best_n:
                best_idx, best_n = i, n
    return best_idx

# === Definición de lectura de archivos TextGrid === #

# Esta función lee un archivo TextGrid y extrae los intervalos de tiempos de inicio y fin de los fonemas.

# Primero carga el archivo TextGrid usando la funcion "TextGrid.fromFile" y utiliza la definicion "textgrid_tier_phoneme_finder"
# para encontrar el nivel (tier) donde estan los fonemas y sus intervalos temporales. En caso de no encontrar el tier, notifica
# al usuario al respecto.

# Si lo encuentra, recorre todos los intervalos del tier seleccionado y:
# Obtiene la etiqueta fonémica (iv.mark) (ej. "a", "k").
# Obtiene el tiempo inicial (iv.minTime) y el tiempo final (iv.maxTime) en segundos (s).
# Descarta los intervalos sin etiqueta (vacíos o silencios definidos en SIL_LABELS).

def textgrid_reader(textgrid_path: str) -> List[Tuple[str, float, float]]:
    SIL_LABELS = {"", "sil", "sp", "spn", "pau"}
    tg = TextGrid.fromFile(textgrid_path)
    idx = textgrid_tier_phoneme_finder(tg)
    if idx is None:
        raise ValueError(f"No phones-like tier found in {textgrid_path}")
    tier = tg.tiers[idx]
    out = []
    for itv in tier.intervals:
        label = (itv.mark or "").strip()
        if label.lower() in SIL_LABELS:
            continue
        start = float(itv.minTime)
        end = float(itv.maxTime)
        if end > start:
            out.append((label, start, end))
    return out

# === Definición de segmentación por fonema === #

# El DTW generalmente se utiliza para piezas de audios completas. Sin embargo, gracias a los archivos TextGrid, es posible
# segmentar los fonemas (label), dando el tiempo inicial (t0) y final (t1) para hacer un análisis espectro-temporal en cada
# fonema de una misma palabra.

# Esta definición, por tanto, recibe una lista de intervalos fonémicos [(label, t0, t1)] ya filtrados (sin silencios) y
# devuelve el intervalo temporal a nivel de palabra, el cual será usado con el algoritmo DTW.

# Si la lista está vacía, retorna (0.0, 0.0) para evitar errores en etapas posteriores.

def phoneme_segmentation(intervals: List[Tuple[str, float, float]]) -> Tuple[float, float]:
    if not intervals:
        return 0.0, 0.0
    return intervals[0][1], intervals[-1][2]

# === Definición de segmentación por palabra === #

# Esta definición retorna los límites globales de un TextGrid en segundos, tal como están almacenados en el archivo.
# Es decir, que define el tiempo de inicio (minTime) y final (maxTime) de una palabra completa para hacer un análisis
# espectro-temporal sin tomar en cuenta la segmentación de fonemas.

def textgrid_full_span(textgrid_path: str) -> tuple[float, float]:
    tg = TextGrid.fromFile(textgrid_path)
    return float(tg.minTime), float(tg.maxTime)

# ========================================================= #
# Definiciones para el algoritmo Dynamic Time Warping (DTW) #
# ========================================================= #

# === Definición de conversión de tiempo (s) a índices de cuadro (frame) === #

# Convierte un intervalo de tiempo real (en segundos (s)) a índices de cuadros en la matriz de características
# (como la matriz MFCC), que es necesario para que sea compatible con las funciones del DTW de la librería "librosa".
# Cada cuadro corresponde a un salto en segundos tomando en cuenta la Frecuencia de muestreo ("SR") y el Desplazamiento
# de ventana ("HOP_LENGTH").

# Primero, se hace la conversión del tiempo inicial (t0) en segundos (s), redondeado hacia abajo (floor) para asegurar que el
# cuadro inicial no quede después del tiempo real.
# Después, se hace la conversión del tiempo final (t1) en segundos (s), redondeado hacia arriba (ceil) para cubrir completamente
# el intervalo.
# Finalmente, se ajustan los valores para que siempre estén dentro de los límites válidos ([0, total_frames]) asegurando que el
# intervalo incluya al menos un cuadro (end >= start + 1).

# Con esto se obtiene el índice del primer cuadro (start) y el índice final + 1 (end).

def time_to_frames(t0: float, t1: float, total_frames: int) -> Tuple[int, int]:
    start = int(math.floor(t0 * SR / HOP_LENGTH))
    end = int(math.ceil(t1 * SR / HOP_LENGTH))
    start = max(0, min(start, max(0, total_frames - 1)))
    end = max(start + 1, min(end, total_frames))
    return start, end

# === Definición de carga de audios === #

# Carga los audios a librosa, asegurándose que tengan la frecuencia de muestreo (SR) y tipo de mezcla (Mono) apropiada.
# En caso de haber un error con algún archivo de audio, se especifica cuál está teniendo problemas.

def load_mono(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if not np.any(np.isfinite(y)):
        raise ValueError(f"Audio appears invalid or empty: {path}")
    return y

# === Definición de espectrograma Mel, conversión logarítmica y transformación a MFCC === #

# La extracción de los costos DTW para grabaciones de habla, se utilizó el siguiente procedimiento:

# Primero, se aplica la STFT (FFT por ventanas) a la señal para obtener una representación espectral de corto tiempo. 
# Estas magnitudes pasan por bancos de filtros en escala Mel (es decir, el espectrograma Mel) para aproximar la percepción humana.
# El resultado son las energías por banda de Mel en el tiempo (S). Esto se hace con la función "librosa.feature.melspectrogram".

# Después, se convierten las energías por banda de Mel (S) a escala logarítmica (dB) para comprimir el rango dinámico y hacer las
# diferencias energéticas más perceptuales. Esto se realiza con la función "librosa.power_to_db", usando la referencia
# "ref=np.max" para normalizar.

# Finalmente, se aplica la Transformada Discreta del Coseno (DCT) sobre las energías logarítmicas para obtener los coeficientes
# MFCC con la funcion librosa.feature.mfcc".

def mfcc_features(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=WINDOW, center=True, power=2.0,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    M = librosa.power_to_db(S, ref=np.max)
    MFCC = librosa.feature.mfcc(S=M, n_mfcc=N_MFCC)
    return MFCC

# === Definición de alineación por Dynamic Time Warping (DTW) === #

# El proceso de alineación de dos señales por DTW requiere de la función "librosa.sequence.dtw" en dos variables,
# la señal base (A) y la señal a comparar (B) empleando métricas computacionales seleccionadas (DTW_METRIC).
# Al usar "backtrack=True", no solo se calcula la matriz acumulada de costos (_), sino que también se reconstruye la ruta
# de alineación óptima (wp), es decir, los pasos que muestran qué muestras de A se alinean con cuáles de B.
# Este último paso es necesario para realizar los procesos de computación BC/BD.

def dtw_path(A: np.ndarray, B: np.ndarray) -> list:
    _, wp = librosa.sequence.dtw(X=A, Y=B, metric=DTW_METRIC, backtrack=True)
    return wp

# === Definición de padding === #

# En casos donde no hay suficientes cuadros para hacer una alineación con DTW, se extrae un subconjunto de cuadros de la matriz
# MFCC entre dos índices de tiempo [start:end], y asegura que el segmento tenga una mínima cantidad de cuadros (MIN_FRAMES).

# Para hacerlo, se extraen los cuadros de la matriz en el rango [start:end] y se mide la longitud del segmento (raw_len).
# Si el segmento es demasiado corto (< MIN_FRAMES), se repite el último cuadro hasta alcanzar la longitud mínima.
# Y si el segmento está vacío (raw_len == 0), se toma el último cuadro válido cercano, y ese será el que se repita
# hasta alcanzar la longitud mínima.

# Finalmente, el conteo de padding presente (padded) que siempre empieza en False, pasa a True si se requiere hacer uso de
# esta definición. Esto permite tener un conteo de cuantos cuadros requirieron de padding para el análisis.

def padding(F: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, int, bool]:
    seg = F[:, start:end]
    raw_len = seg.shape[1]
    padded = False
    if raw_len < MIN_FRAMES:
        if raw_len == 0:
            nearest = min(start, F.shape[1]-1)
            seg = F[:, nearest:nearest+1]
            raw_len = 1
        last = seg[:, -1:].repeat(MIN_FRAMES - raw_len, axis=1)
        seg = np.concatenate([seg, last], axis=1)
        padded = True
    return seg, raw_len, padded


# =================================================== #
# Definiciones para los procesos de computación BC/BD #
# =================================================== #

# === Definición de inicialización de modelo ASR === #

# Se inicializa la configuración y modelo necesaria para el proceso de evaluación BC/BD.
# En este caso, se busca en la librería de "torchaudio" la configuración ASR wav2vec2-CTC y después se descarga el modelo.
# Adicionalmente, se memoriza un caché para acelerar la inicialización y uso de la configuración.

@lru_cache(maxsize=1)
def load_w2v2():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
    return bundle, model

# === Definición de transformación de forma de onda a matriz de posteriorgrama === #

# Carga el modelo ASR previamente inicializado y toma los audios recibidos, asegurándose de que estén a una frecuencia de
# muestreo de 16 kHz y en mono en caso de que no lo estén, con la función "torchaudio.transforms.Resample".

# Finalizado este proceso, la información del audio se transforma de un array Numpy 1D (16000,) a un tensor de 2D (1, 160000)
# con la función "torch.tensor.unsqueeze(0)". Esto es necesario debido a que el modelo requiere una entrada de un tensor 2D
# definido de la siguiente forma: (tamaño del lote, numero de muestras)
# Donde el tamaño del lote siempre es 1, dado que se está procesando un archivo.

# Después, se aplica el modelo ASR al tensor 2D (wav), obteniendo un tensor 3D (emissions).
# El modelo evalúa los aspectos acústicos de los frames de los audios individualmente a través de un proceso similar a la
# cross-correlación, asignando una distribución probabilística a cada uno basado en la base de datos del modelo.

# Finalmente, se normaliza la distribución probabilística con la función "torch.softmax", transponiendo la forma de tal
# forma que termine siendo una matriz de posteriorgrama (C × Tₑ), donde cada columna es una distribución de
# probabilidad de clases (C) en el tiempo (Tₑ), cuya suma de columnas da 1.

def posteriorgram_w2v2(y: np.ndarray, sr: int) -> np.ndarray:
    _, model = load_w2v2()
    if sr != 16000:
        res = torchaudio.transforms.Resample(sr, 16000)
        wav = res(torch.tensor(y).float()).unsqueeze(0)
    else:
        wav = torch.tensor(y).float().unsqueeze(0)
    with torch.no_grad():
        emissions, _ = model(wav)
        post = torch.softmax(emissions[0], dim=-1).T
    return post.cpu().numpy()

# === Definición de renormalización de columnas === #

# Mantiene la coherencia de todas las columnas de la matriz de posteriorgrama, asegurándose que cada paso de tiempo (columna)
# tenga una distribución de probabilidad válida tras la interpolación o el remuestreo. Esto se logra dividiendo cada columna
# entre su propio total, garantizando que cada una sume 1 y evitando divisiones por cero mediante eps.

def renorm_cols(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.sum(P, axis=0, keepdims=True)
    return P / np.clip(s, eps, None)

# === Definición de remuestreo temporal de las matrices de posteriorgrama === #

# Una vez se tengan las matrices de posteriorgrama, es importante alinear el número de columnas del posteriorgrama (P) con el
# conteo de cuadros (frames) MFCC requerido por el algoritmo DTW. Por lo tanto, se toma el conteo (T_src) de la forma de P y se
# compara con el contro de los frames que serán procesados por el DTW (target_T).

# Si ambos conteos coinciden, entonces retorna P, de lo contrario, se construye un mapeo de índices temporales y se redondea al
# vecino más cercano. Después, se duplica o salta columnas según corresponda. Finalmente, se renomraliza cada columna para que
# sume 1 usando renorm_cols(out). Esto garantiza que el número de columnas y el número de frames sean los mismos.

def resample_post_to_mfcc_frames(P: np.ndarray, target_T: int) -> np.ndarray:
    C, T_src = P.shape
    if T_src == target_T:
        return P
    idx = np.clip(np.round(np.linspace(0, T_src - 1, num=target_T)).astype(int), 0, T_src - 1)
    out = P[:, idx]
    return renorm_cols(out)

# === Definición de computación BC/BD === #

# Se hace la computación del promedio del coeficiente de similitud Bhattacharyya (BC) entre las secuencias de las matrices de
# posteriorgrama de la grabación nativa (i) con las del audio sintético (j) siguiendo la ruta de alineación temporal del
# algoritmo DTW (wp).

# Primero, se asegura de que wp tenga contenido interno, o retornará un valor par (0.0, ∞).
# Después, se toma la sumatoria de todas las puntuaciones de la distribución de probabilidad los dos posteriorgramas (P y Q)
# de cada frame i y j, que se irán sumando al coeficiente BC (bcs), asegurándose con "eps" de evitar valores inválidos (log(0)).

# Con esto se obtiene el valor final promedio del coeficiente Bhattacharyya (mean_bc), y de la distancia Bhattacharyya (BD)
# (mean_bd), ambos promediados sobre la ruta (wp).

def bc_bd_compute(P: np.ndarray, Q: np.ndarray, wp, eps: float = 1e-12) -> tuple[float, float]:
    if wp is None or len(wp) == 0:
        return 0.0, float("inf")
    bcs = []
    for i, j in wp:
        b = float(np.sum(np.sqrt(P[:, i] * Q[:, j])))
        bcs.append(max(b, eps))
    mean_bc = float(np.mean(bcs))
    mean_bd = float(np.mean([-math.log(b) for b in bcs]))
    return mean_bc, mean_bd

# ======================= #
# Definiciones de soporte #
# ======================= #

# === Definición de creación de directorios === #

# Se asegura que todos los directorios necesarios estén presentes, o los crea de no estarlos.

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ===================================================== #
# Definición de procesamiento de archivos a gran escala #
# ===================================================== #

def process_item(folder_path: str, name: str):

    # El código espera una carpeta cuyo nombre sea el mismo de los archivos internos. Dentro de esta carpeta se esperan:
    # Un archivo con la grabación real, que será la base ("<nombre> - Real.wav")
    # Un archivo con el audio sintético, que será el archivo a alinear ("<nombre> - TTS.wav")
    # Dos archivos de separación fonética para cada archivo ("<nombre> - Real.TextGrid" y "<nombre> - TTS.TextGrid")

    real_wav = os.path.join(folder_path, f"{name} - Real.wav")
    real_tg  = os.path.join(folder_path, f"{name} - Real.TextGrid")
    tts_wav  = os.path.join(folder_path, f"{name} - TTS.wav")
    tts_tg   = os.path.join(folder_path, f"{name} - TTS.TextGrid")

    # Si falta algún archivo, se notificará cuál es.

    for p in (real_wav, real_tg, tts_wav, tts_tg):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing expected file: {p}")

    # Se crea la carpeta de salida.

    out_dir = os.path.join(OUTPUT_ROOT, name)
    ensure_dir(out_dir)

    # Se cargan los audios.

    y_real = load_mono(real_wav, SR)
    y_tts  = load_mono(tts_wav, SR)

    # Se aplica el espectrograma Mel, la conversión logarítmica y la transformación a MFCC a ambos audios.

    mfcc_real = mfcc_features(y_real)
    mfcc_tts  = mfcc_features(y_tts)

    # Se hace la transformación de forma de onda a matriz de posteriorgrama y se aplica el remuestreo temporal de las matrices.

    post_real_full = resample_post_to_mfcc_frames(posteriorgram_w2v2(y_real, SR), mfcc_real.shape[1])
    post_tts_full = resample_post_to_mfcc_frames(posteriorgram_w2v2(y_tts, SR), mfcc_tts.shape[1])

    # Se realizan las lecturas de los archivos TextGrid.

    intervals_real = textgrid_reader(real_tg)
    intervals_tts  = textgrid_reader(tts_tg)

    # ========================= #
    # Procesamiento de palabras #
    # ========================= #
    
    # Se crea el archivo .CSV donde se guardaran los resultados. En este archivo hay cuatro columnas:
    # Palabra (word)
    # Modo (mode)
    # Coeficiente de similitud Bhattacharyya (bc)
    # Distancia Bhattacharyya (bd)

    word_csv = os.path.join(out_dir, "word_metrics.csv")
    with open(word_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["word", "mode", "bc", "bd"])

        # === Procesamiento de palabras completas === #

        # Se segmentan las palabras usando los archivos TextGrid para determinar el inicio y final de la palabra.

        r_start_sec, r_end_sec = phoneme_segmentation(intervals_real)
        t_start_sec, t_end_sec = phoneme_segmentation(intervals_tts)

        # Se hace la conversión de tiempo (s) a índices de cuadro (frame) basado en el tiempo definido por los archivos TextGrid.
        # Si el tiempo de inicio es menor o igual al del final, se entiende que el archivo TextGrid es inválido, y se utilizaran
        # los frames calculados por la definición del espectrograma Mel, la conversión logarítmica y la transformación a MFCC,
        # tanto para el audio real ("r_start_f, r_end_f") como el sintetico ("t_start_f, t_end_f").

        if r_end_sec <= r_start_sec:
            r_start_f, r_end_f = 0, mfcc_real.shape[1]
        else:
            r_start_f, r_end_f = time_to_frames(r_start_sec, r_end_sec, mfcc_real.shape[1])

        if t_end_sec <= t_start_sec:
            t_start_f, t_end_f = 0, mfcc_tts.shape[1]
        else:
            t_start_f, t_end_f = time_to_frames(t_start_sec, t_end_sec, mfcc_tts.shape[1])

        # Se aplica la definición del padding en casos donde no hay suficientes cuadros para hacer una alineación con DTW.
        # Específicamente se aplica a los segmentos MFCC, que son necesarios para el algoritmo DTW.

        r_seg_full, _, _ = padding(mfcc_real, r_start_f, r_end_f)
        t_seg_full, _, _ = padding(mfcc_tts,  t_start_f, t_end_f)

        # Se aplica la alineación de las dos señales por DTW para obtener la ruta de alineación óptima (wp).

        wp_full = dtw_path(r_seg_full, t_seg_full)

        # Se aplica la definición del padding a la matriz de posteriorgrama.
        # Esto es necesario porque la computación BC/BD funciona con esta matriz, no con los segmentos MFCC directamente.
        # Este padding permite que la matriz y los segmentos tengan al menos la misma cantidad mínima de frames (MIN_FRAMES),
        # y se pueda utilizar la ruta de alineación óptima (wp) del DTW acordemente.

        r_post_full, _, _ = padding(post_real_full, r_start_f, r_end_f)
        t_post_full, _, _ = padding(post_tts_full,  t_start_f, t_end_f)
        P_full = r_post_full
        Q_full = t_post_full

        # Se aplica la definición de computación BC/BD para poder obtener el Coeficiente Bhattacharyya (BC), y de la
        # Distancia Bhattacharyya (BD).

        bc_full, bd_full = bc_bd_compute(P_full, Q_full, wp_full)

        # Se escribe en el documento .CSV los datos:
        # Palabra (name)
        # Modo de palabras completas ("full")
        # El Coeficiente Bhattacharyya (bc_full)
        # La Distancia Bhattacharyya (bd_full)

        writer.writerow([name, "full", f"{bc_full:.6f}", f"{bd_full:.6f}"])

        # === Procesamiento de palabras por fonemas === #

        # Primero se define la cantidad total de intervalos a trabajar (n_pairs) basado en los archivos TextGrid.
        # Se toma en cuenta la cantidad total de ambos archivos y se toma el menor número de intervalos entre los dos para que
        # se puedan realizar los cálculos. Idealmente, ambos archivos deberían tener la misma cantidad de intervalos.

        n_pairs = min(len(intervals_real), len(intervals_tts))

        # Se definen las variables que se van a utilizar:
        # Contador de fonemas (n_used)
        # Sumatoria de coeficientes Bhattacharyya (bc_sum)
        # Sumatoria de distancias Bhattacharyya (bd_sum)

        n_used = 0
        bc_sum = 0.0
        bd_sum = 0.0
        
        # Se realizan operaciones en todos los intervalos encontrados (n_pairs).

        for i in range(n_pairs):

            # Se obtienen los intervalos de tiempo del intervalo [i].

            r_lab, r_t0, r_t1 = intervals_real[i]
            t_lab, t_t0, t_t1 = intervals_tts[i]

            # Se realizan las conversiones de tiempo (s) a índices de cuadro (frame)..

            r_s, r_e = time_to_frames(r_t0, r_t1, mfcc_real.shape[1])
            t_s, t_e = time_to_frames(t_t0, t_t1, mfcc_tts.shape[1])

            # Se aplica la definición del padding en casos donde no hay suficientes cuadros para hacer una alineación con DTW.

            r_seg, r_raw, _ = padding(mfcc_real, r_s, r_e)
            t_seg, t_raw, _ = padding(mfcc_tts,  t_s, t_e)

            # Se aplica la alineación de las dos señales por DTW para obtener la ruta de alineación óptima (wp).

            wp_i = dtw_path(r_seg, t_seg)

            # Se aplica la definición del padding a la matriz de posteriorgrama.

            r_post_seg, _, _ = padding(post_real_full, r_s, r_e)
            t_post_seg, _, _ = padding(post_tts_full,  t_s, t_e)
            P_i = r_post_seg
            Q_i = t_post_seg

            # Se aplica la definición de computación BC/BD.

            bc_i, bd_i = bc_bd_compute(P_i, Q_i, wp_i)

            # Los resultados son aplicados a las variables de sumatoria de coeficiente (bc_sum) y distancia (bd_sum).

            bc_sum += bc_i
            bd_sum += bd_i

            # Se incrementa el contador de fonemas (n_used).

            n_used += 1

        # Se hace la sumatoria final de coeficientes (bc_agg) y distancias (bd_agg) siendo divididas por el número contador
        # de fonemas (que definen cuantas operaciones fueron realizadas). En caso de que no se haya analizado ningún fonema
        # (n_used == 0), se confirma que ambos audios no tienen ninguna relación, indicando potencialmente un error.

        if n_used == 0:
            bc_agg, bd_agg = 0.0, float("inf")
        else:
            bc_agg = bc_sum / n_used
            bd_agg = bd_sum / n_used

        # Se escribe en el documento .CSV los datos:
        # Palabra (name)
        # Modo de de palabras por fonemas ("per_per_phone")
        # Media aritmética de Coeficiente Bhattacharyya (bc_agg)
        # Media aritmética de Distancia Bhattacharyya (bd_full)

        writer.writerow([name, "per_phone", f"{bc_agg:.6f}", f"{bd_agg:.6f}"])

    # ======================== #
    # Procesamiento de fonemas #
    # ======================== #
    
    # Se crea el archivo .CSV donde se guardaran los resultados. En este archivo hay catorce columnas:
    # Palabra (word)
    # Número del index del fonema (idx)
    # Fonema real (real_label)
    # Tiempo de inicio de fonema real (real_t0)
    # Tiempo final de fonema real (real_t1)
    # Cantidad de frames de fonema real (real_frames_raw)
    # Cantidad de frames que fueron agregados por padding del fonema real (real_padded)
    # Fonema sintético (tts_label)
    # Tiempo de inicio de fonema sintético (tts_t0)
    # Tiempo final de fonema sintético (tts_t1)
    # Cantidad de frames de fonema sintético (tts_frames_raw)
    # Cantidad de frames que fueron agregados por padding del fonema sintético (tts_padded)
    # Coeficiente de similitud Bhattacharyya (bc)
    # Distancia Bhattacharyya (bd)

    phone_csv = os.path.join(out_dir, "phoneme_metrics.csv")
    with open(phone_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["word","idx",
            "real_label","real_t0","real_t1","real_frames_raw","real_padded",
            "tts_label","tts_t0","tts_t1","tts_frames_raw","tts_padded",
            "bc","bd"
        ])

        # Se realiza un proceso idéntico al procesamiento de palabras por fonemas, solo que no se realiza una sumatoria
        # al final para determinar la media aritmética de los coeficientes y la distancia, si no que se toma nota de los
        # resultados individuales de cada fonema.

        n_pairs = min(len(intervals_real), len(intervals_tts))

        for i in range(n_pairs):
            r_lab, r_t0, r_t1 = intervals_real[i]
            t_lab, t_t0, t_t1 = intervals_tts[i]

            r_s, r_e = time_to_frames(r_t0, r_t1, mfcc_real.shape[1])
            t_s, t_e = time_to_frames(t_t0, t_t1, mfcc_tts.shape[1])

            r_seg, r_raw, r_pad = padding(mfcc_real, r_s, r_e)
            t_seg, t_raw, t_pad = padding(mfcc_tts,  t_s, t_e)

            wp = dtw_path(r_seg, t_seg)

            r_post, _, _ = padding(post_real_full, r_s, r_e)
            t_post, _, _ = padding(post_tts_full,  t_s, t_e)
            P = r_post
            Q = t_post

            bc, bd = bc_bd_compute(P, Q, wp)

            writer.writerow([
                name, i,
                r_lab, f"{r_t0:.6f}", f"{r_t1:.6f}", r_raw, int(r_pad),
                t_lab, f"{t_t0:.6f}", f"{t_t1:.6f}", t_raw, int(t_pad),
                f"{bc:.6f}", f"{bd:.6f}",
            ])

# =================== #
# Definición del main #
# =================== #

def main():

    # Se filtran las notificaciones de "UserWarning" para determinar con menos información el comportamiento del código.

    warnings.filterwarnings("ignore", category=UserWarning)

    # Se utiliza o crea el directorio de salida del código.

    ensure_dir(OUTPUT_ROOT)

    # Se notifica con un print si el directorio de entrada no existe (o tiene otro nombre).

    if not os.path.isdir(INPUT_ROOT):
        raise FileNotFoundError(f'Input root "{INPUT_ROOT}" not found.')

    # Se realiza todo el procedimiento del código, notificando qué directorio se está procesando y, si fue un éxito,
    # se muestra el directorio de salida de una forma conveniente. El directorio de salida siempre creará directorios con
    # el nombre de los directorios dentro del directorio de entrada, de forma que sea más claro saber que cálculos le pertenecen
    # a qué par. En caso de un error, se notifica que directorio tuvo un error y porque.

    for entry in sorted(os.listdir(INPUT_ROOT)):
        folder_path = os.path.join(INPUT_ROOT, entry)
        if not os.path.isdir(folder_path):
            continue
        name = entry
        try:
            print(f"[INFO] Procesando: {name}")
            process_item(folder_path, name)
            print(f"[OK]   Resultados -> {os.path.join(OUTPUT_ROOT, name)}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

# ========================= #
# Inicialización del codigo #
# ========================= #

if __name__ == "__main__":
    main()
