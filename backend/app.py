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
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json # Explicitly import for handling GOOGLE_CREDENTIALS_JSON
import httpx

# Import Telegram bot setup from our new module (relative import)
# This assumes 'telegram_bot_handlers.py' is in the same directory as 'app.py'
from . import telegram_bot_handlers
from telegram import Update

# NEW IMPORTS for News Feed (PostgreSQL)
import psycopg2
import psycopg2.extras # For DictCursor
from urllib.parse import urlparse
from datetime import datetime, timedelta
import feedparser

load_dotenv()

# --- Load Environment Variables using the EXACT names you provided ---
OPENAI_API_KEY = os.getenv("OPENAI API KEY")
GEMINI_API_KEY = os.getenv("GEMINI API KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM BOT TOKEN")
# Convert comma-separated string to a list of integers
AUTHORIZED_TELEGRAM_USER_IDS = [int(x) for x in os.getenv("AUTHORIZED TELEGRAM USER IDS", "").split(',') if x.strip().isdigit()]
DATABASE_URL = os.getenv("DATABASE URL")
RENDER_SERVICE_URL = os.getenv("RENDER SERVICE URL")
GOOGLE_CREDENTIALS_JSON_CONTENT = os.getenv("GOOGLE CREDENTIALS JSON") # This holds the JSON string content

# --- FastAPI App Setup ---
app = FastAPI()

# Allow CORS for all origins during development. Restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting this to your frontend domain in production (e.g., ["https://your-app.onrender.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Google Text-to-Speech Client Initialization (using GOOGLE_CREDENTIALS_JSON) ---
tts_client = None # Initialize as None, will be set up in startup event

@app.on_event("startup")
async def startup_event():
    # Initialize the PostgreSQL database for news feeds
    init_news_db()

    # Setup Google TTS credentials from environment variable
    global tts_client # Declare global to modify the tts_client variable
    if GOOGLE_CREDENTIALS_JSON_CONTENT:
        try:
            # Create a temporary file to store the credentials
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, "google_credentials.json")

            with open(temp_file_path, "w") as temp_file:
                # Ensure the content is valid JSON before writing
                credentials_data = json.loads(GOOGLE_CREDENTIALS_JSON_CONTENT)
                json.dump(credentials_data, temp_file)

            # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable for the current process
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
            tts_client = texttospeech.TextToSpeechClient() # Initialize TTS client
            # print("Google Text-to-Speech client initialized successfully.") # Removed for no logs

        except json.JSONDecodeError:
            raise RuntimeError("GOOGLE_CREDENTIALS_JSON environment variable is not valid JSON.")
        except Exception as e:
            raise RuntimeError(f"Error setting up Google Text-to-Speech credentials from environment variable: {e}")
    else:
        # This means GOOGLE_CREDENTIALS_JSON was not set or empty. TTS will likely fail.
        # Attempt to initialize anyway, but it's likely to fail unless ADC is otherwise configured.
        try:
            tts_client = texttospeech.TextToSpeechClient()
        except Exception as e:
            raise RuntimeError(f"Google_credentials_json not found, and TTS client initialization failed: {e}")

    # The Telegram bot local polling is handled in the __main__ block for local dev
    # On Render, it's driven by webhooks, so no polling setup needed here.
    pass # This startup event already handles db init and TTS client setup

# --- Async Function for Text-to-Speech ---
async def text_to_speech(text: str) -> bytes:
    if not tts_client:
        # Handle case where TTS client failed to initialize
        # print("Text-to-Speech client not initialized. Cannot synthesize speech.") # Removed for no logs
        return b"" # Return empty bytes or raise a specific error

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-C", # Or another suitable voice
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
            model="gpt-4o", # Or gpt-3.5-turbo if you prefer
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

        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Construct the parts for Gemini
        parts = []
        if caption:
            parts.append({"text": f"Fact-check this image based on the caption: '{caption}'"})
        else:
            parts.append({"text": "Fact-check this image. Is there anything misleading or false within the image content itself? Provide a concise factual analysis."})

        parts.append({
            "inline_data": {
                # Attempt to guess MIME type, fallback to common image/jpeg
                "mime_type": imghdr.what(None, h=image_bytes) or mimetypes.guess_type('image.jpg')[0],
                "data": base64_image
            }
        })

        data = {
            "contents": [{
                "parts": parts
            }]
        }

        try:
            gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
            response = await client.post(gemini_url, headers=headers, json=data, timeout=30.0) # Add timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            response_json = response.json()
            return response_json['candidates'][0]['content']['parts'][0]['text']
        except httpx.HTTPStatusError as e:
            return f"Error contacting fact-checking service: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Network error contacting fact-checking service: {e}"
        except Exception as e:
            return f"An unexpected error occurred during image fact-check: {e}"

# --- API Endpoints ---

# Serves index.html directly from the root.
# This assumes index.html is in the same directory as app.py.
@app.get("/")
async def read_root():
    return FileResponse("index.html")

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
        return JSONResponse(status_code=500, content={"error": "Internal server error during text fact-check"})

@app.post("/factcheck/audio")
async def fact_check_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        # Use tempfile for handling audio, ensure cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
        
        # Open the file again for OpenAI API
        with open(temp_audio_file.name, "rb") as audio_file_for_openai:
            transcription_response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_for_openai,
                response_format="text"
            )
        transcription = transcription_response
        os.unlink(temp_audio_file.name) # Clean up temp file

        if not transcription:
            return JSONResponse(status_code=500, content={"error": "Failed to transcribe audio"})

        fact_check_result = await get_text_fact_check_response(transcription)
        audio_result_b64 = None
        if fact_check_result:
            audio_bytes_tts = await text_to_speech(fact_check_result)
            audio_result_b64 = base64.b64encode(audio_bytes_tts).decode('utf-8')

        return JSONResponse(content={"transcription": transcription, "result": fact_check_result, "audio_result": audio_result_b64})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal server error during audio fact-check"})

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
        return JSONResponse(status_code=500, content={"error": "Internal server error during image fact-check"})

# --- Telegram Webhook ---
# Ensure telegram_bot_handlers.py is accessible at the same level as app.py
telegram_application = None # Initialize as None
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
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- NEWS FEED SECTION (PostgreSQL Integration) ---

class AddSourceRequest(BaseModel):
    url: str
    name: str

# Database connection function for PostgreSQL
def get_news_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE URL environment variable not set. News Feed functionality will not work without a database connection.")

    result = urlparse(DATABASE_URL)
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port if result.port else 5432 # Default PostgreSQL port

    conn = psycopg2.connect(
        database=database,
        user=username,
        password=password,
        host=hostname,
        port=port
    )
    return conn

# Database initialization for PostgreSQL
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

# This ensures database and TTS client are initialized when FastAPI starts
@app.on_event("startup")
async def startup_event_handler(): # Renamed to avoid conflict with global startup_event
    await startup_event() # Call the main startup_event logic

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
            # Use DictCursor to easily access columns by name
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
                        continue # Move to the next source

                    # If not cached or cache expired, fetch fresh
                    feed = feedparser.parse(source_url)

                    # Handle bozo feeds gracefully (parsing errors)
                    if feed.bozo:
                        pass # No logging as per instruction

                    if not feed.entries:
                        # Add a placeholder for sources with no entries
                        all_articles.append({
                            "title": f"No articles available from {source_name}",
                            "link": "#",
                            "source": source_name,
                            "pubDate": None
                        })
                        continue

                    # Clear old articles for this source before inserting new ones
                    cur.execute("DELETE FROM articles WHERE source_id = %s", (source_id,))

                    for entry in feed.entries:
                        title = entry.title
                        link = entry.link
                        pub_date_str = None

                        # Attempt to get a robust published date in ISO format
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                pub_date_str = datetime(*entry.published_parsed[:6]).isoformat() + "Z"
                            except (TypeError, ValueError):
                                pass # Fallback if datetime conversion fails

                        # If parsing failed, try raw published attribute or updated
                        if not pub_date_str:
                            pub_date_str = getattr(entry, 'published', getattr(entry, 'updated', None))

                        all_articles.append({
                            "title": title,
                            "link": link,
                            "source": source_name,
                            "pubDate": pub_date_str
                        })

                        # Insert into cache, handle potential duplicates
                        try:
                            cur.execute("INSERT INTO articles (source_id, title, link, pub_date, fetched_at) VALUES (%s, %s, %s, %s, %s)",
                                        (source_id, title, link, pub_date_str, datetime.now().isoformat()))
                            conn.commit()
                        except psycopg2.errors.UniqueViolation:
                            conn.rollback() # Rollback current transaction if duplicate link exists
                        except Exception:
                            conn.rollback()

            # Sort all articles by date, newest first. Handle cases where pubDate might be None.
            # Articles with None pubDate will be placed at the very end (datetime.min)
            all_articles.sort(key=lambda x: datetime.fromisoformat(x['pubDate'].replace('Z', '')) if x['pubDate'] else datetime.min, reverse=True)

            return JSONResponse(content={"articles": all_articles}, status_code=200)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch news articles: {e}"})

# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))

    # This block handles local Telegram bot polling.
    # It will be skipped when RENDER_SERVICE_URL is set (i.e., on Render deployment).
    if TELEGRAM_BOT_TOKEN and not RENDER_SERVICE_URL:
        # The telegram_application needs to be initialized here for local polling to work.
        # Ensure telegram_application is defined in this scope.
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
        # This block runs the FastAPI app, for Render deployment or if Telegram bot is not configured locally.
        uvicorn.run(app, host="0.0.0.0", port=port)