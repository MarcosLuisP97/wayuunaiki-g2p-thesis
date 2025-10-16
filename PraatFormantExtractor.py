import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parselmouth as pm

# ========================================================= #
# Parámetros para el analisis Praat a traves de ParselMouth #
# ========================================================= #

# Se definen los parámetros que utilizara la librería "ParselMouth" para hacer los análisis. Estos parámetros son:
    # Paso de tiempo, en segundos (s) (TIME_STEP).
    # La frecuencia máxima del rango de búsqueda de formantes, en hercios (Hz) (FORMANT_CEILING).
    # La duración efectiva de la ventana de análisis, en segundos (s) (WINIDOW_LENGTH).
    # El punto de +3 dB para un filtro de paso bajo invertido con una pendiente de +6 dB/octava (PRE_EMPHASIS).
    # Los últimos cuatro parámetros son modificables al inicio del código.
# En este caso, se seleccionaron los parámetros por defecto que se encuentran en el software "Praat"

FORMANT_CEILING = 5500
WINIDOW_LENGTH   = 0.025
MAX_N_FORMANTS = 5
PRE_EMPHASIS = 50
TIME_STEP = 0

# ========================================================================== #
# Parámetros para la graficación de los resultados del análisis de formantes #
# ========================================================================== #

# Se definen los parametros que se utilizaran para la graficación de los resultados del análisis de formantes:
# F2 es el axis X con su punto minimo (F2_MIN) y su punto maximo (F2_MAX).
# F1 es el axis Y con su punto minimo (F1_MIN) y su punto maximo (F1_MAX).
# Adicionalmente se define el relleno vertical (en Hz) que se utiliza para colocar cada etiqueta de vocal ligeramente alejada
# de su punto medio en el eje F1 para que el texto no se superponga al marcador (LABEL_Y_OFFSET).

F2_MIN, F2_MAX = 900, 2950
F1_MIN, F1_MAX = 250, 1300
LABEL_Y_OFFSET = 20

# =========================================================== #
# Definiciones de lectura y organizacion de archivos TextGrid #
# =========================================================== #

# === Definición de identificación de vocales === #

# Primero se define la lista de caracteres que serán considerados vocales, con el propósito de delimitar la extracción de
# formantes a solamente este tipo de fonemas.

VOWELS = set(list("aAeEiIoOuUáéíóúÁÉÍÓÚɨɯ"))

def is_vowel_label(lbl: str) -> bool:
    if not lbl: return False
    return lbl.strip()[0] in VOWELS

# === Definición de lectura de archivos TextGrid === #

# Se lee un archivo TextGrid de la misma forma que se haría con la función "Read from file..." del software Praat.
# En caso de que cualquier error en la lectura, se le notificara al usuario.

def load_textgrid(pm, path: Path):
    try:
        tg = pm.praat.call("Read from file...", str(path))
        return tg
    except Exception as e:
        print(f"[INFO] Parselmouth 'Read from file...' failed for {path.name}: {e}", file=sys.stderr)

# === Definición de buscador de nivel (tier) de fonemas en archivos TextGrid === #

# Los archivos TextGrid pueden tener múltiples niveles (tiers) o estar configurados de formas distintas. Esta definición ayuda a
# a buscar nivel (tier) donde específicamente se encuentran los fonemas y sus intervalos temporales.

# Inicialmente, se extrae el número de tiers en el archivo (n), y después se extrae el nombre de los tiers de cada uno (name).
# Si se encuentra el tier bajo el nombre "phones" (que es el nombre más común para tier donde se encuentran los fonemas), se
# retorna ese tier.

def get_phones_tier_index(pm, tg_obj):
    n = int(pm.praat.call(tg_obj, "Get number of tiers"))
    for i in range(1, n+1):
        name = str(pm.praat.call(tg_obj, "Get tier name...", i)).strip().lower()
        if name == "phones":
            return i
    return None

# ===================================================== #
# Definición de extracción de formantes con ParselMouth #
# ===================================================== #

def extract_formants_for_pair(snd, tg_obj, pm, tier_idx: int) -> pd.DataFrame:
    
    # Inicialmente, se extrae él numera de intervalos/segmentaciones encontradas en el tier de fonemas definido previamente.

    n_intervals = int(pm.praat.call(tg_obj, "Get number of intervals...", tier_idx))

    # Los parámetros de Praat que utilizara ParselMouth para la extracción de formantes. En este orden los parámetros son:
    # Tipo de archivo a analizar (snd).
    # Rastreador de formantes LPC de Praat (Burg).
    # Paso de tiempo, en segundos (s) (TIME_STEP).
    # La frecuencia máxima del rango de búsqueda de formantes, en hercios (Hz) (FORMANT_CEILING).
    # La duración efectiva de la ventana de análisis, en segundos (s) (WINIDOW_LENGTH).
    # El punto de +3 dB para un filtro de paso bajo invertido con una pendiente de +6 dB/octava (PRE_EMPHASIS).
    # Los últimos cuatro parámetros son modificables al inicio del código.

    fm = pm.praat.call(snd, "To Formant (burg)",
                       float(TIME_STEP), float(MAX_N_FORMANTS), float(FORMANT_CEILING), float(WINIDOW_LENGTH), float(PRE_EMPHASIS))

    # Se crea un index donde se guardaran los fonemas analizados (label), y los primeros tres fonemas en hercios (Hz)
    # (f1m, f2m, f3m).

    rows = []

    # Se analiza todos los fonemas encontrados en los intervalos/segmentaciones (n_intervals).

    for k in range(1, n_intervals + 1):

        # Se leen los caracteres dentro del intervalo, asegurándose de que sea una vocal con la definición previa.

        label = str(pm.praat.call(tg_obj, "Get label of interval...", tier_idx, k)).strip()
        if not label or not is_vowel_label(label):
            continue

        # Se obtienen los tiempos de inicio (t1) y final (t2) del fonema, con el cual se obtiene también la duración.
        # Si la duración es igual o menor a 0, se entiende que es un intervalo inválido y se salta.

        t1 = float(pm.praat.call(tg_obj, "Get start time of interval...", tier_idx, k))
        t2 = float(pm.praat.call(tg_obj, "Get end time of interval...", tier_idx, k))
        dur = t2 - t1
        if dur <= 0.0:
            continue

        # Se extraen la primera (f1m), segunda (f2m) y tercera (f3m) formante acordemente utilizando la función de ParselMouth,
        # delimitado por los tiempos calculados.

        f1m = pm.praat.call(fm, "Get mean...", 1, t1, t2, "Hertz")
        f2m = pm.praat.call(fm, "Get mean...", 2, t1, t2, "Hertz")
        f3m = pm.praat.call(fm, "Get mean...", 3, t1, t2, "Hertz")
        
        # Se escriben los resultados en el index.

        rows.append({
            "phone": label,
            "F1_Hz": f1m,
            "F2_Hz": f2m,
            "F3_Hz": f3m,
        })

    # Finalmente, se guardan todos los resultados de todas las vocales en la variable (df).

    df = pd.DataFrame(rows)

    # En caso de números negativos, se añade un procesamiento final que garantiza que solo se retornaran valores positivos.

    if not df.empty:
        df = df[(df["F1_Hz"] > 0) & (df["F2_Hz"] > 0)]
    return df

# ======================= #
# Definiciones de soporte #
# ======================= #

# === Definición de agrupación de datos === #

# Al finalizar la extracción de formantes individuales, se realiza el cálculo agrupando todas las vocales.
# Si la variable de datos en la definición previa está vacía, se retorna las primeras columnas que dictan el título de los datos.
# En caso de que se encuentre la misma vocal múltiples veces en la misma palabra, se describen cuantas vocales fueron encontradas
# (N), y se obtiene la media aritmética de la primera (F1_mean), segunda (F2_mean) y tercera formante (F3_mean).

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["phone","F1_mean","F2_mean","F3_mean","N"])
    return (df.groupby("phone").agg(F1_mean=("F1_Hz", "mean"), F2_mean=("F2_Hz", "mean"), F3_mean=("F3_Hz", "mean"), N=("phone", "size")).reset_index())

# === Definición de límites para el plot === #

# Se aplican los límites para los axis de la primera (xlim) y la segunda formante (ylim), definidas al principio del código.

def apply_fixed_axes():
    plt.xlim(F2_MIN, F2_MAX)
    plt.ylim(F1_MIN, F1_MAX)  

# ================================================ #
# Definición de escritura de archivos .CSV y plots #
# ================================================ #

def save_outputs(df: pd.DataFrame, out_dir: Path, prefix: str):

    # Primero, se lee o se crea el directorio de salida.

    out_dir.mkdir(parents=True, exist_ok=True)

    # Después, se realiza el análisis de formantes individual y se crea el archivo .CSV.

    csv_path = out_dir / f"{prefix} formants.csv"
    df.to_csv(csv_path, index=False)

    # Después, se realiza el análisis de formantes agrupados y se crea el archivo .CSV.

    summary = summarize(df)
    summary_path = out_dir / f"{prefix} formants_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Finalmente, se hace la graficación de las formantes extraídas
    # Las vocales tienen una forma y color definido en el plot.
    # Si hay múltiples vocales en la misma palabra, se muestran los puntos donde se encontraron, y la media aritmética se vuelve
    # el punto más definido.

    plt.figure(figsize=(8, 5))
    plt.scatter(df["F2_Hz"], df["F1_Hz"], alpha=0.25, s=20)
    for _, row in summary.iterrows():
        plt.scatter(row["F2_mean"], row["F1_mean"], s=70)
        plt.text(row["F2_mean"], row["F1_mean"] - LABEL_Y_OFFSET, row["phone"],
                    fontsize=12, fontweight="bold", ha="center", va="top")
    plt.xlabel("F2 (Hz)")
    plt.ylabel("F1 (Hz)")
    plt.title(f"Vowel Space with Per-Vowel Means — {prefix}")
    apply_fixed_axes()
    png_path = out_dir / f"{prefix} vowel_space_labeled.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

# ================================================================================== #
# Definición de escritura de archivos .CSV y plots de los resultados de ambos audios #
# ================================================================================== #

def save_combined_outputs(real_df: pd.DataFrame, tts_df: pd.DataFrame, out_dir: Path):

    # Primero, se lee o se crea el directorio de salida

    out_dir.mkdir(parents=True, exist_ok=True)

    # Después, se realiza el análisis de formantes agrupados de ambos audios.

    real_sum = summarize(real_df)
    tts_sum  = summarize(tts_df)

    # Después, se crea el archivo .CSV teniendo en cuenta los resultados de ambos audios y calculando la diferencia y distancia
    # Euclidiana de las formantes, así también como tomando en cuenta el número de vocales encontradas.

    r = real_sum.rename(columns={"F1_mean":"Real_F1_mean","F2_mean":"Real_F2_mean","F3_mean":"Real_F3_mean","N":"Real_N"})
    t = tts_sum .rename(columns={"F1_mean":"TTS_F1_mean","F2_mean":"TTS_F2_mean","F3_mean":"TTS_F3_mean","N":"TTS_N"})
    merged = pd.merge(r[["phone","Real_F1_mean","Real_F2_mean","Real_F3_mean","Real_N"]],
                      t[["phone","TTS_F1_mean","TTS_F2_mean","TTS_F3_mean","TTS_N"]],
                      on="phone", how="outer")
    merged["dF1_Hz"] = merged["TTS_F1_mean"] - merged["Real_F1_mean"]
    merged["dF2_Hz"] = merged["TTS_F2_mean"] - merged["Real_F2_mean"]
    merged["dF3_Hz"] = merged["TTS_F3_mean"] - merged["Real_F3_mean"]
    merged["euclid_Hz"] = (merged["dF1_Hz"]**2 + merged["dF2_Hz"]**2) ** 0.5
    cols = ["phone","Real_F1_mean","Real_F2_mean","Real_F3_mean","Real_N",
            "TTS_F1_mean","TTS_F2_mean","TTS_F3_mean","TTS_N",
            "dF1_Hz","dF2_Hz","dF3_Hz","euclid_Hz"]
    merged = merged[[c for c in cols if c in merged.columns]]
    (out_dir / "COMBINED formants_summary.csv").write_text(merged.to_csv(index=False))

    # Finalmente, se crea un plot, siguiendo la misma lógica que la definición previa.

    plt.figure(figsize=(9, 5.5))
    if not real_df.empty:
        plt.scatter(real_df["F2_Hz"], real_df["F1_Hz"], alpha=0.25, s=18, label="Real intervals")
    if not tts_df.empty:
        plt.scatter(tts_df["F2_Hz"], tts_df["F1_Hz"], alpha=0.25, s=18, label="TTS intervals")
    for _, row in real_sum.iterrows():
        plt.scatter(row["F2_mean"], row["F1_mean"], s=80, marker="o", label=None)
        plt.text(row["F2_mean"], row["F1_mean"] - LABEL_Y_OFFSET, row["phone"], fontsize=12, fontweight="bold", ha="center", va="top")
    for _, row in tts_sum.iterrows():
        plt.scatter(row["F2_mean"], row["F1_mean"], s=80, marker="s", label=None)
        plt.text(row["F2_mean"], row["F1_mean"] - LABEL_Y_OFFSET, row["phone"], fontsize=12, fontweight="bold", ha="center", va="top")
    plt.xlabel("F2 (Hz)"); plt.ylabel("F1 (Hz)")
    plt.title("Vowel Space with Per-Vowel Means — COMBINED (Real ○, TTS □)")
    apply_fixed_axes(); plt.legend(loc="best")
    path = out_dir / "COMBINED vowel_space_labeled.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

# ======================================= #
# Definición de procesamiento de archivos #
# ======================================= #

def process_pair(word_dir: Path, out_word_dir: Path, word_name: str, kind: str, pm):

    # Inicialmente, se confirma la existencia de los archivos .WAV y TextGrid con sus debidos nombres.

    wav = word_dir / f"{word_name} - {kind}.wav"
    tg  = word_dir / f"{word_name} - {kind}.TextGrid"

    # Después, se define el archivo de sonido, archivo TextGrid, y el tier del archivo TextGrid que se va a usar para la
    # extracción de formantes.

    snd = pm.Sound(str(wav))
    tg_obj = load_textgrid(pm, tg)
    tier_idx = get_phones_tier_index(pm, tg_obj)

    # Despues, se realiza la extracción de formantes.

    df = extract_formants_for_pair(snd, tg_obj, pm, tier_idx)

    # En caso de que no se haya podido encontrar el tier adecuado o si los resultados de todas las vocales en la variable (df)
    # estan vacios, se salta el procedamiento por completo.

    if tier_idx is None:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    # Finalmente, se realiza el proceso de creacion de archivos .CSV y su plot.

    save_outputs(df, out_word_dir, prefix=kind)

    return df

# =================== #
# Definición del main #
# =================== #

def main():

    # Primero, se define el directorio base, y los directorios de entrada y salida.
    
    base = Path(__file__).resolve().parent
    in_root  = base / "Praat Entrada"
    out_root = base / "Praat Salida"
    out_root.mkdir(parents=True, exist_ok=True)

    # Después, se leen todos los subdirectorios del directorio de entrada para hacer el procedimiento a gran escala.

    word_dirs = [p for p in in_root.iterdir() if p.is_dir()]

    # Por cada palabra encontrada en los diretorios, se realiza el procedimeinto completo.

    for word_dir in sorted(word_dirs):

        # Se extrae el nombre del directorio, el cual será utilizara para crear un subdirectorio en el directorio de salida,
        # permitiendo simetría entre ambos directorios.

        word_name = word_dir.name
        out_word_dir = out_root / word_name
        print(f"\n=== Procesando palabra: {word_name} ===")

        # Después se realizan los procesos individuales de las grabaciones reales y los audios sintéticos, finalizando con
        # el procesamiento final de ambos.

        real_df = process_pair(word_dir, out_word_dir, word_name, "Real", pm)
        tts_df  = process_pair(word_dir, out_word_dir, word_name, "TTS", pm)
        save_combined_outputs(real_df, tts_df, out_word_dir)

        print(f"=== Palabra procesada:  {word_name} ===")

    print(f"\n=== Procesamiento completo. Revisar directorio: {out_root} ===")

# ========================= #
# Inicialización del codigo #
# ========================= #

if __name__ == "__main__":
    main()