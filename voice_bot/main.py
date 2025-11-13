import os
import platform
import speech_recognition as sr
from groq import Groq
from dotenv import load_dotenv
import ffmpeg
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
templates = Jinja2Templates(directory="voice_bot/templates")
recognizer = sr.Recognizer()

# ✅ Only initialize TTS locally
engine = None
if platform.system() == "Windows":
    import pyttsx3
    engine = pyttsx3.init()


def speak_text(text):
    """Speak only if local environment supports it"""
    if engine:
        engine.say(text)
        engine.runAndWait()
    else:
        print(f"[TTS skipped: server mode] Bot: {text}")


def generate_response(prompt: str):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a friendly, reflective AI answering interview-style questions , ignore asterisks ,dash ,dots,slashes and other special characters in the response.keep the response concise and to the point."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process-voice/")
async def process_audio(file: UploadFile = File(...)):
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_in:
            temp_in_path = temp_in.name
            temp_in.write(await file.read())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            temp_out_path = temp_out.name

        # Convert webm → wav
        (
            ffmpeg
            .input(temp_in_path)
            .output(temp_out_path, format='wav', ar='16000', ac=1)
            .overwrite_output()
            .run(quiet=True)
        )

        with sr.AudioFile(temp_out_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        response = generate_response(text)
        speak_text(response)

        os.remove(temp_in_path)
        os.remove(temp_out_path)

        return JSONResponse({"text": text, "reply": response})
    except Exception as e:
        return JSONResponse({"error": str(e)})


