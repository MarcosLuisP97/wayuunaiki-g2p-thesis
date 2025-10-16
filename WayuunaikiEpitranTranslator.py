import os
import subprocess
import epitran

# =================================================================================== #
# Definición de transcripción de texto Wayuunaiki a simbología AFI utilizando Epitran #
# =================================================================================== #

# Se importa el modelo/diccionario Wayuunaiki y se aplica al texto. 

def transcribe_to_ipa_wayuunaiki(text):

    # El modelo debe estar en los archivos internos de Epitran, como si fuese otro modelo más.

    epi = epitran.Epitran('way-Latn')
    return epi.transliterate(text)

# ============================================================================ #
# Definición de mapeo manual de simbología AFI a texto compatible con Festival #
# ============================================================================ #

# Definición de mapeo manual de simbología AFI a texto compatible con Festival.
# Traduce símbolos AFI a letras cuyos sonidos en español sean los más similares a Wayuunaiki.

def map_ipa_to_festival_wayuunaiki(ipa_text):

    IPA_TO_FESTIVAL_WAYUU = {

        # Lista de mapeo manual de simbología AFI a texto compatible con Festival.
        # En algunos casos, por problemas de interacciones con Epitran, se deben hacer mapeos aún con letras
        # compatibles en español para que se suenen similares a los fonemas Wayuunaiki.

        "ɯ": "u",
        "ʃ": "sh",
        "ʔ": "'",
        "ɺ": "r",
        "aː": "aa",
        "eː": "ee",
        "iː": "ii",
        "oː": "oo",
        "nː": "n",
        "uː": "uu",
        "ɯː": "üü"
    }

    for ipa, festival in IPA_TO_FESTIVAL_WAYUU.items():
        ipa_text = ipa_text.replace(ipa, festival)
    return ipa_text

# ================================================================= #
# Definición de exportación de audio artificial a formato de sonido #
# ================================================================= #

# Se define el nombre del archivo a exportar en "output_file" para conveniencia del proceso.
# Esto se debe a que el software Praat no lee archivos cuyos nombres tienen caracteres especiales como "ü".

def synthesize_with_festival_wayuunaiki(ipa_text, output_file="asiipuu.wav"):

    # Se crea un archivo de texto temporal con el texto que Festival "lee" en formato UTF-8.

    temp_file = "temp_ipa_wayuunaiki.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(ipa_text)

    # Se corre el proceso text2wave con el modelo de voz en español "(voice_el_diphone)".
    # Finalmente, se borra el archivo de texto temporal.

    try:
        subprocess.run(
            ["text2wave", "-eval", "(voice_el_diphone)", temp_file, "-o", output_file],
            check=True,
        )
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return output_file

# Proceso central del código, donde se ingresa el texto Wayuunaiki en "text_wayuunaiki".
# Adicionalmente, se hace un print de todos los procesos para observar todos los procesos de traducción.

if __name__ == "__main__":

    text_wayuunaiki = "Asiipüü" # Ejemplo
    print(f"Texto Wayuunaiki: {text_wayuunaiki}")

    ipa_result = transcribe_to_ipa_wayuunaiki(text_wayuunaiki)
    print(f"Transcripción a simbología AFI: {ipa_result}")

    festival_text = map_ipa_to_festival_wayuunaiki(ipa_result)
    print(f"Transcripción a texto compatible con Festival: {festival_text}")

    output_file = synthesize_with_festival_wayuunaiki(festival_text)
    print(f"Audio artificialha sido guardado como: {output_file}")
