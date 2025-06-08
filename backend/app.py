import os
import io
import base64
import tempfile
import mimetypes
import imghdr
from pydub import AudioSegment
from google.cloud import texttospeech
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles # NEW IMPORT
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json
import httpx

from . import telegram_bot_handlers
from telegram import Update

# NEW IMPORTS for News Feed (PostgreSQL)
import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
from datetime import datetime, timedelta
import feedparser

load_dotenv()

# --- Load Environment Variables using standard underscore naming ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_TELEGRAM_USER_IDS = [int(x) for x in os.getenv("AUTHORIZED_TELEGRAM_USER_IDS", "").split(',') if x.strip().isdigit()]
DATABASE_URL = os.getenv("DATABASE_URL")
RENDER_SERVICE_URL = os.getenv("RENDER_SERVICE_URL")
GOOGLE_CREDENTIALS_JSON_CONTENT = os.getenv("GOOGLE_CREDENTIALS_JSON")

# --- FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Google Text-to-Speech Client Initialization ---
tts_client = None

# --- Determine the path to the 'frontend' directory ---
# This correctly navigates from app.py (inside backend/) up to the project root,
# and then down into the frontend/ directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # This is /opt/render/project/src/backend/
PROJECT_ROOT = os.path.join(BASE_DIR, "..")          # This is /opt/render/project/src/
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend") # This is /opt/render/project/src/frontend/

# Mount static files (CSS, JS, images, etc.) from the 'frontend' directory
# These files will be served at paths starting with /static/
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.on_event("startup")
async def startup_event():
    init_news_db()

    global tts_client
    if GOOGLE_CREDENTIALS_JSON_CONTENT:
        try:
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, "google_credentials.json")
            with open(temp_file_path, "w") as temp_file:
                credentials_data = json.loads(GOOGLE_CREDENTIALS_JSON_CONTENT)
                json.dump(credentials_data, temp_file)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
            tts_client = texttospeech.TextToSpeechClient()
        except json.JSONDecodeError:
            raise RuntimeError("GOOGLE_CREDENTIALS_JSON environment variable is not valid JSON.")
        except Exception as e:
            raise RuntimeError(f"Error setting up Google Text-to-Speech credentials from environment variable: {e}")
    else:
        try:
            tts_client = texttospeech.TextToSpeechClient()
        except Exception as e:
            raise RuntimeError(f"GOOGLE_CREDENTIALS_JSON not found, and TTS client initialization failed: {e}")
    pass

# --- Async Function for Text-to-Speech ---
async def text_to_speech(text: str) -> bytes:
    if not tts_client:
        return b""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-C",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content

# --- OpenAI and Gemini API for Fact-Checking ---
async def get_text_fact_check_response(text: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI fact-checker. Analyze the given text for accuracy and provide a concise, factual summary or a verdict of true/false/misleading with a brief explanation. If you cannot determine, state that. Prioritize factual accuracy."},
                {"role": "user", "content": f"Fact-check this: {text}"}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during text fact-checking: {e}"

async def perform_image_factcheck(image_bytes: bytes, caption: str = None) -> str:
    async with httpx.AsyncClient() as client:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        parts = []
        if caption:
            parts.append({"text": f"Fact-check this image based on the caption: '{caption}'"})
        else:
            parts.append({"text": "Fact-check this image. Is there anything misleading or false within the image content itself? Provide a concise factual analysis."})
        parts.append({
            "inline_data": {
                "mime_type": imghdr.what(None, h=image_bytes) or mimetypes.guess_type('image.jpg')[0],
                "data": base64_image
            }
        })
        data = {"contents": [{"parts": parts}]}
        try:
            gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
            response = await client.post(gemini_url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            response_json = response.json()
            return response_json['candidates'][0]['content']['parts'][0]['text']
        except httpx.HTTPStatusError as e:
            return f"Error contacting fact-checking service: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Network error contacting fact-checking service: {e}"
        except Exception as e:
            return f"An unexpected error occurred during image fact-check: {e}"

# --- API Endpoints ---

# This route will serve your index.html file when someone accesses the root URL.
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/factcheck/text")
async def fact_check_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
        if not text:
            return JSONResponse(status_code=400, content={"error": "No text provided"})
        result = await get_text_fact_check_response(text)
        audio_result_b64 = None
        if result:
            audio_bytes = await text_to_speech(result)
            audio_result_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return JSONResponse(content={"result": result, "audio_result": audio_result_b64})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error during text fact-check: {e}"})

@app.post("/factcheck/audio")
async def fact_check_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
        with open(temp_audio_file.name, "rb") as audio_file_for_openai:
            transcription_response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_for_openai,
                response_format="text"
            )
        transcription = transcription_response
        os.unlink(temp_audio_file.name)
        if not transcription:
            return JSONResponse(status_code=500, content={"error": "Failed to transcribe audio"})
        fact_check_result = await get_text_fact_check_response(transcription)
        audio_result_b64 = None
        if fact_check_result:
            audio_bytes_tts = await text_to_speech(fact_check_result)
            audio_result_b64 = base64.b64encode(audio_bytes_tts).decode('utf-8')
        return JSONResponse(content={"transcription": transcription, "result": fact_check_result, "audio_result": audio_result_b64})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error during audio fact-check: {e}"})

@app.post("/factcheck/image")
async def fact_check_image(file: UploadFile = File(...), caption: str = Form(None)):
    try:
        image_bytes = await file.read()
        result = await perform_image_factcheck(image_bytes, caption)
        audio_result_b64 = None
        if result:
            audio_bytes = await text_to_speech(result)
            audio_result_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return JSONResponse(content={"result": result, "audio_result": audio_result_b64})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error during image fact-check: {e}"})

# --- Telegram Webhook ---
telegram_application = None
if TELEGRAM_BOT_TOKEN:
    telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
    telegram_bot_handlers.initialize_bot_components(
        openai_client_instance=openai_client,
        tts_func=text_to_speech,
        authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
        perform_image_factcheck_func=perform_image_factcheck
    )

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    if not TELEGRAM_BOT_TOKEN or not telegram_application:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content="Telegram bot not configured.")
    try:
        req_json = await request.json()
        update = Update.de_json(req_json, telegram_application.bot)
        if update:
            await telegram_application.process_update(update)
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        print(f"Error processing Telegram webhook: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- NEWS FEED SECTION (PostgreSQL Integration) ---

class AddSourceRequest(BaseModel):
    url: str
    name: str

def get_news_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set. News Feed functionality will not work without a database connection.")
    result = urlparse(DATABASE_URL)
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port if result.port else 5432
    conn = psycopg2.connect(
        database=database,
        user=username,
        password=password,
        host=hostname,
        port=port
    )
    return conn

def init_news_db():
    try:
        with get_news_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS sources (
                        id SERIAL PRIMARY KEY,
                        url TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS articles (
                        id SERIAL PRIMARY KEY,
                        source_id INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        link TEXT NOT NULL UNIQUE,
                        pub_date TEXT,
                        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
                    )
                ''')
                conn.commit()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PostgreSQL database: {e}")

@app.on_event("startup")
async def startup_event_handler():
    await startup_event()

@app.post("/news/add_source")
async def add_news_source(source_data: AddSourceRequest):
    try:
        with get_news_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO sources (url, name) VALUES (%s, %s)", (source_data.url, source_data.name))
            conn.commit()
        return JSONResponse(content={"message": "News source added successfully!"}, status_code=200)
    except psycopg2.errors.UniqueViolation:
        return JSONResponse(content={"error": "This news source URL already exists."}, status_code=409)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to add news source: {e}"})

@app.get("/news/articles")
async def get_news_articles():
    all_articles = []
    try:
        with get_news_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT id, url, name FROM sources")
                sources = cur.fetchall()

                for source in sources:
                    source_id = source['id']
                    source_name = source['name']
                    source_url = source['url']

                    one_hour_ago = datetime.now() - timedelta(hours=1)
                    cur.execute(
                        "SELECT title, link, pub_date FROM articles WHERE source_id = %s AND fetched_at > %s",
                        (source_id, one_hour_ago.isoformat())
                    )
                    cached_articles = cur.fetchall()

                    if cached_articles:
                        for article in cached_articles:
                            all_articles.append({
                                "title": article['title'],
                                "link": article['link'],
                                "source": source_name,
                                "pubDate": article['pub_date']
                            })
                        continue

                    feed = feedparser.parse(source_url)
                    if feed.bozo:
                        pass

                    if not feed.entries:
                        all_articles.append({
                            "title": f"No articles available from {source_name}",
                            "link": "#",
                            "source": source_name,
                            "pubDate": None
                        })
                        continue

                    cur.execute("DELETE FROM articles WHERE source_id = %s", (source_id,))

                    for entry in feed.entries:
                        title = entry.title
                        link = entry.link
                        pub_date_str = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                pub_date_str = datetime(*entry.published_parsed[:6]).isoformat() + "Z"
                            except (TypeError, ValueError):
                                pass
                        if not pub_date_str:
                            pub_date_str = getattr(entry, 'published', getattr(entry, 'updated', None))
                        all_articles.append({
                            "title": title,
                            "link": link,
                            "source": source_name,
                            "pubDate": pub_date_str
                        })
                        try:
                            cur.execute("INSERT INTO articles (source_id, title, link, pub_date, fetched_at) VALUES (%s, %s, %s, %s, %s)",
                                        (source_id, title, link, pub_date_str, datetime.now().isoformat()))
                            conn.commit()
                        except psycopg2.errors.UniqueViolation:
                            conn.rollback()
                        except Exception:
                            conn.rollback()

            all_articles.sort(key=lambda x: datetime.fromisoformat(x['pubDate'].replace('Z', '')) if x['pubDate'] else datetime.min, reverse=True)
            return JSONResponse(content={"articles": all_articles}, status_code=200)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch news articles: {e}"})

# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))

    if TELEGRAM_BOT_TOKEN and not RENDER_SERVICE_URL:
        if 'telegram_application' not in locals() or telegram_application is None:
            telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
            telegram_bot_handlers.initialize_bot_components(
                openai_client_instance=openai_client,
                tts_func=text_to_speech,
                authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
                perform_image_factcheck_func=perform_image_factcheck
            )
        telegram_application.run_polling(poll_interval=3.0)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)