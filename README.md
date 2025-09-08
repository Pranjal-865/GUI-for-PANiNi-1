# GUI-for-PANiNi-1
#this will work with cheifnetizens code
import customtkinter as ctk
import threading
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import argostranslate.package
import argostranslate.translate
from kittentts import KittenTTS
import soundfile as sf
from playsound import playsound

# -----------------------------
# Initialize TTS
# -----------------------------
tts = KittenTTS("KittenML/kitten-tts-nano-0.2")

# -----------------------------
# Whisper model (tiny = fastest for Pi4)
# -----------------------------
model = whisper.load_model("tiny")

# -----------------------------
# Recording parameters
# -----------------------------
fs = 24000  # sample rate
duration = 5  # seconds

# -----------------------------
# CustomTkinter GUI setup
# -----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("320x240")  # 2.8 inch screen resolution
app.title("Offline Translator")

# -----------------------------
# Dropdown for language selection
# -----------------------------
languages = ["Hindi", "Telugu", "English"]
selected_lang = ctk.StringVar(value="Hindi")

lang_menu = ctk.CTkOptionMenu(app, values=languages, variable=selected_lang)
lang_menu.pack(pady=20)

# -----------------------------
# Output label
# -----------------------------
output_label = ctk.CTkLabel(app, text="Translation will appear here", wraplength=280)
output_label.pack(pady=20)

# -----------------------------
# Core function: Record → STT → Translate → TTS
# -----------------------------
def run_translation():
    output_label.configure(text="Recording...")
    app.update()

    # Record from mic
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write("input.wav", fs, recording)

    # Transcribe with Whisper
    audio = whisper.load_audio("input.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(language="hi", task="transcribe")
    result = whisper.decode(model, mel, options)

    spoken_text = result.text
    print("Recognized:", spoken_text)

    # Argos translation (if needed)
    from_code = {"Hindi": "hi", "Telugu": "te", "English": "en"}[selected_lang.get()]
    to_code = "en"

    if from_code != "en":
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages)
        )
        argostranslate.package.install_from_path(package_to_install.download())
        translated_text = argostranslate.translate.translate(spoken_text, from_code, to_code)
    else:
        translated_text = spoken_text  # English already

    print("Translated:", translated_text)

    # Update GUI
    output_label.configure(text=translated_text)

    # Speak output
    audio_out = tts.generate(translated_text, voice="expr-voice-5-m")
    sf.write("speak.wav", audio_out, 24000)
    playsound("speak.wav")

# -----------------------------
# Button to start process
# -----------------------------
translate_button = ctk.CTkButton(app, text="Translate to English", command=lambda: threading.Thread(target=run_translation).start())
translate_button.pack(pady=20)

# -----------------------------
# Run app
# -----------------------------
app.mainloop()
