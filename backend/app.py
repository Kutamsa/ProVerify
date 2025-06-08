from google.cloud import texttospeech
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import tempfile
import os
import uvicorn
from openai import OpenAI
import base64
from dotenv import load_dotenv
import json

# Import Telegram bot setup from our new module
from . import telegram_bot_handlers # Relative import within backend package
from telegram import Update # Import Update class directly here for de_json

load_dotenv()

# --- IMPORTANT ---
# Ensure your OPENAI_API_KEY is set in your .env file locally,
# and as an environment variable on Render.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- TELEGRAM BOT CONFIG ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# Authorized user IDs for bot access (comma-separated string in env var)
# Example: AUTHORIZED_TELEGRAM_USER_IDS="12345,67890"
AUTHORIZED_TELEGRAM_USER_IDS = [int(uid) for uid in os.getenv("AUTHORIZED_TELEGRAM_USER_IDS", "").split(',') if uid.strip().isdigit()]

app = FastAPI()

# Serve static frontend files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Serve index.html at root URL
@app.get("/")
def serve_index():
    index_file_path = os.path.join(frontend_path, "index.html")
    return FileResponse(index_file_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

# --- GOOGLE TTS CREDENTIALS HANDLING ---
google_credentials_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON")

if google_credentials_json_str:
    try:
        temp_gcloud_key_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_gcloud_key_file.write(google_credentials_json_str.encode('utf-8'))
        temp_gcloud_key_file.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_gcloud_key_file.name
        print(f"Google Cloud credentials loaded from GOOGLE_CREDENTIALS_JSON environment variable to: {temp_gcloud_key_file.name}")
    except Exception as e:
        print(f"Error processing GOOGLE_CREDENTIALS_JSON environment variable: {e}")
else:
    print("GOOGLE_CREDENTIALS_JSON environment variable not found. Attempting local fallback for Google TTS.")
    try:
        local_key_path = os.path.join(os.path.dirname(__file__), "gcloud_key.json")
        if os.path.exists(local_key_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_key_path
            print(f"Using local Google Cloud credentials from: {local_key_path}")
        else:
            print(f"Error: Local gcloud_key.json not found at {local_key_path}. Google TTS will likely not work.")
    except Exception as e:
        print(f"Error setting local GOOGLE_APPLICATION_CREDENTIALS: {e}")

tts_client = texttospeech.TextToSpeechClient()

def text_to_speech(text: str) -> str:
    """Converts text to speech using Google Cloud TTS and returns base64 encoded MP3 audio."""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="te-IN",
        name="te-IN-Chirp3-HD-Achird",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return base64.b64encode(response.audio_content).decode('utf-8')

# --- FACT-CHECKING FUNCTIONS (reused and now directly called by web app and bot) ---
async def perform_text_factcheck(input_text: str):
    prompt = f"Fact-check this text and provide a clear answer:\n\n{input_text}"
    ai_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a helpful fact-checking assistant. Respond clearly and concisely."},
            {"role": "user", "content": prompt}
        ]
    )
    fact_check_result = ai_response.choices[0].message.content
    audio_base64 = text_to_speech(fact_check_result)
    return {"result": fact_check_result, "audio_result": audio_base64}

async def perform_audio_factcheck(audio_file_path: str):
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        transcribed_text = transcript.text

    prompt = f"Fact-check this audio content (transcribed in Telugu):\n\n{transcribed_text}"
    ai_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a Telugu fact-checking assistant. Provide clear and concise answers in Telugu."},
            {"role": "user", "content": prompt}
        ]
    )
    fact_check_result = ai_response.choices[0].message.content
    audio_base64_result = text_to_speech(fact_check_result)
    return {"transcription": transcribed_text, "result": fact_check_result, "audio_result": audio_base64_result}

async def perform_image_factcheck(image_bytes: bytes, mime_type: str, caption: str):
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {"role": "system", "content": "You are a fact-checking assistant. Provide clear and concise answers."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": caption or "Please fact-check this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                }
            ],
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500
    )
    return {"result": response.choices[0].message.content}

# --- WEB APP ENDPOINTS (now call shared fact-checking functions) ---
@app.post("/factcheck/text")
async def factcheck_text_web(input: TextInput):
    try:
        response = await perform_text_factcheck(input.text)
        return response
    except Exception as e:
        print(f"Error in web factcheck_text: {e}")
        return {"error": str(e)}

@app.post("/factcheck/audio")
async def factcheck_audio_web(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        response = await perform_audio_factcheck(tmp_path)
        return response
    except Exception as e:
        print(f"Error in web factcheck_audio: {e}")
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/factcheck/image")
async def factcheck_image_web(
    file: UploadFile = File(...),
    caption: str = Form("")
):
    try:
        image_bytes = await file.read()
        response = await perform_image_factcheck(image_bytes, file.content_type, caption)
        return JSONResponse(content=response)
    except Exception as e:
        print(f"Error in web factcheck_image: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/factcheck/tts")
async def generate_tts_web(input: TextInput):
    try:
        audio_base64 = text_to_speech(input.text)
        return {"audio": audio_base64}
    except Exception as e:
        print(f"Error in web generate_tts: {e}")
        return {"error": str(e)}


# --- TELEGRAM BOT SETUP ---
telegram_application = None # Initialize as None

@app.on_event("startup")
async def startup_event():
    """Initialize bot components and set webhook when FastAPI app starts."""
    global telegram_application # Declare global to assign

    # Initialize bot components in the separate module
    telegram_bot_handlers.initialize_bot_components(
        openai_client_instance=client,
        tts_func=text_to_speech,
        authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
        perform_image_factcheck_func=perform_image_factcheck
    )

    if TELEGRAM_BOT_TOKEN:
        telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
        
        webhook_url = os.getenv("RENDER_EXTERNAL_HOSTNAME")
        if webhook_url:
            full_webhook_url = f"https://{webhook_url}/telegram-webhook"
            print(f"Setting Telegram webhook to: {full_webhook_url}")
            await telegram_application.bot.set_webhook(url=full_webhook_url)
            print("Telegram webhook set successfully.")
        else:
            print("RENDER_EXTERNAL_HOSTNAME not found. Webhook not set for production. Running in local polling mode (if __name__ is called).")
    else:
        print("TELEGRAM_BOT_TOKEN not found. Telegram bot functionality will not be active.")

@app.on_event("shutdown")
async def shutdown_event():
    """Remove the webhook for the Telegram bot when the FastAPI app shuts down."""
    if telegram_application and os.getenv("RENDER_EXTERNAL_HOSTNAME"):
        print("Removing Telegram webhook...")
        await telegram_application.bot.delete_webhook()
        print("Telegram webhook removed.")

# --- TELEGRAM WEBHOOK ENDPOINT ---
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request) -> Response:
    """Handle incoming Telegram updates."""
    if not TELEGRAM_BOT_TOKEN or not telegram_application:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content="Telegram bot not configured.")

    try:
        req_json = await request.json()
        # FIX: Correctly parse the JSON into an Update object
        update = Update.de_json(req_json, telegram_application.bot)
        
        if update: # Check if update object is valid
            await telegram_application.process_update(update)
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        print(f"Error processing Telegram webhook: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- Main execution block ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # When running locally via `python app.py` and no RENDER_EXTERNAL_HOSTNAME,
    # the bot runs in polling mode. Otherwise, Uvicorn runs for the web app and webhook.
    if TELEGRAM_BOT_TOKEN and not os.getenv("RENDER_EXTERNAL_HOSTNAME"):
        print("Running Telegram bot in local polling mode...")
        # This will block, so it's typically for local development
        telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
        # Initialize bot components for local polling too
        telegram_bot_handlers.initialize_bot_components(
            openai_client_instance=client,
            tts_func=text_to_speech,
            authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
            perform_image_factcheck_func=perform_image_factcheck
        )
        telegram_application.run_polling(poll_interval=1)
    else:
        print("Running Uvicorn server for web app and webhook endpoint...")
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

