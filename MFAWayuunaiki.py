import os
import shutil
import subprocess
from pydub import AudioSegment
import epitran

# ====================================================================== #
# Definición de arreglo automatizado de símbolos AFI para compatibilidad #
# ====================================================================== #

# Toma en cuenta el contexto de reglas gramaticales en español para que sean aplicadas a textos Wayuunaiki. 

def fix_phonemes_for_mfa(phoneme_str):

    # Se separan los caracteres (símbolos AFI) para poder ser evaluados como ítems distintos y se define el conteo.

    phonemes = phoneme_str.split()
    fixed = []
    i = 0
    char_index = 0

    # Se evalúa cada carácter (p), y se va defendiendo cuáles son sus dos caracteres siguientes (nxt, nxt2) y su
    # carácter previo (prev) para hacer correcciones acordes.

    while i < len(phonemes):
        p = phonemes[i]
        nxt = phonemes[i + 1] if i < len(phonemes) - 1 else None
        nxt2 = phonemes[i + 2] if i < len(phonemes) - 2 else None

        # === Arreglo /t͡ʃ/ === #
        # El fonema /t͡ʃ/ (el cual es asociado al sonido "ch" en español) oficialmente tiene un arco o barra de unión,
        # por lo que Epitran también lo utiliza. Sin embargo, el diccionario de MFA utiliza el fonema /tʃ/,
        # que también es válido, pero se requiere de hacer esta corrección para que sea compatible.

        if p == 't' and nxt == '͡' and nxt2 == 'ʃ':
            fixed.append('tʃ')
            i += 3
            char_index += 2
            continue

        # === Arreglo de unión extra (t + ʃ → tʃ) === #
        if p == 't' and nxt == 'ʃ':
            fixed.append('tʃ')
            i += 2
            char_index += 2
            continue

        # === Arreglo /ʎ/ a /ʝ/ === #
        # Tradicionalmente en español, palabras con "ll" o "y" usan el fonema /ʎ/ para pronunciarse.
        # Sin embargo, varias regiones de habla hispana utilizan el fonema /ʝ/ ("Yeísmo").
        # Entre estas regiones se encuentra la región de la guajira, donde reside la tribu Wayuunaiki.
        # Por tanto, se hizo este cambio para tomar en cuenta esta particularidad fonética.

        if p == 'ʎ':
            fixed.append('ʝ')
            i += 1
            char_index += 2
            continue

        # === Arreglo /t/ a /t̪/ === #
        # Aunque en español y Wayuunaiki existen los fonemas /t/ y /t̪/ para pronunciar palabras con la letra "t",
        # el diccionario de MFA utiliza solo el fonema /t̪/.
        # Por lo tanto, se ajusta para que todas las palabras con "t" se interpreten con el fonema /t̪/.
        # Adicionalmente, en Wayuunaiki hay múltiples palabras con dos "t" que son fonéticamente idénticas a una "t".
        # Se añadió una lógica para que palabras con dos "t" se igualara a un solo fonema /t̪/.

        elif p == 't'and nxt == 't':
            fixed.append('t̪')
            i += 2
            char_index += 2
            continue
        elif p == 't':
            fixed.append('t̪')

        # === Arreglos /ɺ/ y /ɯ/ === #
        # En español no se utilizan los fonemas /ɺ/ y /ɯ/, sin embargo, auditivamente, son muy similares a los fonemas
        # /ɾ/ y /u/ respectivamente, por lo que se hace este reemplazo para que fuesen compatible con el
        # diccionario de MFA.

        elif p == 'ɺ':
            fixed.append('ɾ')
        elif p == 'ɯ':
            fixed.append('u')

        else:
            fixed.append(p)

        i += 1
        char_index += 1

    # === Eliminación de fonemas /:/, /ʔ/ y /h/ === #
    # En español no se utilizan los fonemas /:/ y /ʔ/, y las palabras con la letra "h" son fonéticamente silenciosas,
    # por lo que estos fonemas son eliminados para que fuesen compatible con el diccionario de MFA.

    fixed = [p for p in fixed if p != "ː"]
    fixed = [p for p in fixed if p != "ʔ"]
    fixed = [p for p in fixed if p != "h"]

    return " ".join(fixed)

# =========== #
# Directorios #
# =========== #

# === Directorio base === #
# Se define el directorio base donde va a operar el código.

base_dir = os.path.dirname(os.path.abspath(__file__))

# === Directorio de entrada === #
# Se utiliza (o crea en caso de no haberlo) el directorio de entrada ("inputs_dir").
# Dentro de este directorio debe estar el archivo de sonido que se busca alinear y un archivo .txt
# El nombre del archivo de sonido, del archivo .txt y el contenido dentro del archivo de sonido debe ser los mismos.
# Si es una palabra o una oración, todos deben compartir el mismo texto, incluyendo mayúsculas, espacios y caracteres especiales.

inputs_dir = os.path.join(base_dir, "MFA Entrada")

# === Directorio de archivos procesados === #
# Se utiliza (o crea en caso de no haberlo) el directorio de archivos procesados ("speaker_dir").
# Dentro de este directorio se integran los archivos de sonido y .txt procesados para ser utilizados por el MFA.
# El MFA requiere que los archivos estén preparados en un directorio llamado "speaker1". Este nombre no se puede cambiar.
# Para conveniencia, si ya existe el directorio "speaker1", esta será borrada y recreada, donde se pondrán nuevos archivos.

speaker_dir = os.path.join(inputs_dir, "speaker1")
if os.path.exists(speaker_dir):
    shutil.rmtree(speaker_dir)
os.makedirs(speaker_dir, exist_ok=True)

# === Directorio de diccionario temporal === #
# Se define el directorio para la creación del diccionario temporal ("custom_dict_path").

custom_dict_path = f"{base_dir}/wayuunaiki_mfa.dict"

# ==================================== #
# Procesamiento de conversión de audio #
# ==================================== #

# MFA requiere que todos los audios estén en formato .WAV PCM de 16 bits en mono. Idealmente a 16kHz.
# Este procesamiento automático ayuda a simplificar el proceso con formatos de audio populares (como MP3).
# Al finalizar el proceso de conversión, crea el directorio de salida de MFA ("output_dir") y dentro de esa carpeta,
# una carpeta adicional con el nombre del archivo de sonido que se procesó.
# Si ya existe una carpeta con ese nombre, será borrada y recreada para conveniencia del usuario.
# También se creará una copia del audio procesado, de forma que el archivo de sonido y el archivo TextGrid estén en el
# mismo directorio y puedan ser usados con Praat más convenientemente.

for filename in os.listdir(inputs_dir):
    if filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
        filepath = os.path.join(inputs_dir, filename)
        sound = AudioSegment.from_file(filepath)
        sound = sound.set_frame_rate(16000).set_channels(1)
        base_name = os.path.splitext(filename)[0]
        base_id = base_name.replace(" ", "_")
        wav_name = base_id + ".wav"
        output_path = os.path.join(speaker_dir, wav_name)
        sound.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        output_dir = os.path.join(base_dir, "MFA Salida", os.path.basename(base_id))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(output_path, os.path.join(output_dir, wav_name))

# ================================== #
# Procesamiento de archivos de texto #
# ================================== #

# MFA requiere que todos los archivos de textos estén en formato UTF-8, y que todas las palabras a analizar estén
# posicionadas en fila. Este procesamiento ayuda a facilitar el proceso en caso de que se busque analizar oraciones.
# Adicionalmente, crea una copia de la palabra al lado de la original, que es necesario para el siguiente proceso.

all_words = set()
for filename in os.listdir(inputs_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(inputs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        words = text.split()
        all_words.update(words)

        base_name = os.path.splitext(filename)[0]
        base_id = base_name.replace(" ", "_")
        output_path = os.path.join(speaker_dir, f"{base_id}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{base_id} {text}\n")

# ========================================= #
# Creación de diccionario temporal para MFA #
# ========================================= #

# El formato de diccionarios de MFA es muy particular, ya que requiere que todas las palabras dentro del audio estén en fila
# y la fonología AFI de cada palabra este en columna, a un espacio de distancia de cada una. Ejemplo:
# abandona	a β a n d̪ o n a
# abandonada	a β a n d̪ o n a ð a
# El proceso anterior permite que cada palabra en fila tenga una copia a su lado. Este proceso permitirá hacer la
# separación de cada letra, hacer la traducción a simbología AFI y crear un diccionario con el mismo formato que los que
# utiliza MFA, que será guardado temporalmente un archivo temporal en formato .dict en el directorio base.
# Esto incluye el proceso visto en "Definición de arreglo automatizado de símbolos AFI para compatibilidad"

with open(custom_dict_path, "w", encoding="utf-8") as dict_file:
    for word in sorted(all_words):

        # Se inicializa Epitran asumiendo que se tiene el modelo/diccionario Wayuunaiki "way-Latn" dentro de su base
        # de datos interna.

        epi = epitran.Epitran("way-Latn")
        ipa = epi.transliterate(word)
        phonemes = fix_phonemes_for_mfa("   ".join(" ".join(ch for ch in ipa).split()))
        dict_file.write(f"{word} {phonemes}\n")

# ======================= #
# Montreal Forced Aligner #
# ======================= #

# Se corre el MFA, definiendo que directorios y funciones va a utilizar.

subprocess.run([
    "mfa", "align", # Selección de libreria (MFA) y función (alinear sonidos).
    os.path.join(speaker_dir), # Directorio de entrada (archivo de sonido y archivo de texto).
    os.path.join(custom_dict_path), # Directorio del diccionario temporal.
    "spanish_mfa", # Selección de modelo acústico entrenado interno (en este caso es el modelo español).
    os.path.join(output_dir), # Directorio de salida (donde sale el archivo .TextGrid).
    "--clean", # Función de clean (para limpiar el cache al terminar cualquier proceso).
    "--single_speaker", "--num_jobs", "1" # Delimitación de hablantes en el audio a procesar (solo un hablante).
], check=True)

# Como paso final, se borra el diccionario temporal creado.

if os.path.exists(custom_dict_path):
    os.remove(custom_dict_path)
