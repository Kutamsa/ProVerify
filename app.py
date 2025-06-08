import os
import io
import base64
import tempfile
import mimetypes
import imghdr
from pydub import AudioSegment
from google.cloud import texttospeech
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles # Ensure this is imported
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json
import httpx
from contextlib import asynccontextmanager

from . import telegram_bot_handlers
from telegram import Update

import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
from datetime import datetime, timedelta
import feedparser

load_dotenv()

# --- Load Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_TELEGRAM_USER_IDS = [int(x) for x in os.getenv("AUTHORIZED_TELEGRAM_USER_IDS", "").split(',') if x.strip().isdigit()]
DATABASE_URL = os.getenv("DATABASE_URL")
RENDER_SERVICE_URL = os.getenv("RENDER_SERVICE_URL")
GOOGLE_CREDENTIALS_JSON_CONTENT = os.getenv("GOOGLE_CREDENTIALS_JSON")

# Initialize OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Google Text-to-Speech Client ---
tts_client = None
temp_google_credentials_path = None # Global variable to store path for cleanup

# FIX 3: Initialize telegram_application globally and robustly
telegram_application = None
if TELEGRAM_BOT_TOKEN:
    try:
        telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
        # Note: tts_func will be set in the lifespan event after tts_client is initialized
        telegram_bot_handlers.initialize_bot_components(
            openai_client_instance=openai_client,
            tts_func=None, # Temporarily None, will be set in lifespan
            authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
            perform_image_factcheck_func=perform_image_factcheck
        )
        print("Telegram bot components initialized globally (partial).")
    except Exception as e:
        print(f"Error initializing Telegram bot components globally: {e}")
        telegram_application = None # Ensure it's None if initialization fails

# --- FastAPI App Setup with Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_client, temp_google_credentials_path, telegram_application

    # Initialize Google TTS client and credentials
    if GOOGLE_CREDENTIALS_JSON_CONTENT:
        try:
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            temp_file.write(GOOGLE_CREDENTIALS_JSON_CONTENT)
            temp_file.close()
            temp_google_credentials_path = temp_file.name # Store path to delete later
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_google_credentials_path
            tts_client = texttospeech.TextToSpeechClient()
            print("Google Text-to-Speech client initialized.")
        except Exception as e:
            print(f"Error initializing Google TTS client: {e}")
            tts_client = None
    else:
        print("GOOGLE_CREDENTIALS_JSON environment variable not set. Google TTS will be unavailable.")

    # Initialize database
    init_news_db()

    # FIX 3: Update telegram_application with the real tts_func after tts_client is ready
    if telegram_application:
        telegram_bot_handlers.initialize_bot_components(
            openai_client_instance=openai_client,
            tts_func=text_to_speech, # Now passing the real text_to_speech function
            authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
            perform_image_factcheck_func=perform_image_factcheck
        )
        print("Telegram bot components updated with TTS function.")

    yield # Application starts here

    # Cleanup on shutdown
    if temp_google_credentials_path and os.path.exists(temp_google_credentials_path):
        os.remove(temp_google_credentials_path)
        print(f"Cleaned up temporary Google credentials file: {temp_google_credentials_path}")

app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files (HTML, CSS, JS) ---
# IMPORTANT CHANGE HERE:
# Mount the 'static' directory explicitly
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Database Initialization (PostgreSQL) ---
def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(DATABASE_URL)

def init_news_db():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                url VARCHAR(2048) NOT NULL UNIQUE,
                last_fetched TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                title VARCHAR(2048) NOT NULL,
                link VARCHAR(2048) NOT NULL UNIQUE,
                pub_date TIMESTAMP WITH TIME ZONE,
                fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("Database tables 'sources' and 'articles' ensured to exist.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

# --- Utility Functions ---
async def text_to_speech(text: str) -> bytes:
    if not tts_client:
        print("TTS client not available.")
        return b""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="te-IN",
        name="te-IN-Chirp3-HD-Achird",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return b""

async def transcribe_audio(audio_file: UploadFile) -> str:
    # Save the uploaded audio to a temporary file
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        content = await audio_file.read()
        temp_audio_file.write(content)
        temp_audio_file.close()

        # OpenAI Whisper API for transcription
        with open(temp_audio_file.name, "rb") as audio:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        return transcript.text
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return f"Error transcribing audio: {e}"
    finally:
        os.remove(temp_audio_file.name)

async def perform_text_factcheck(text: str) -> str:
    # FIX 7: Missing openai_client check
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key is not configured for text fact-checking."
    try:
        chat_completion = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and accurate fact-checking assistant. Provide concise and neutral fact-checks for the given text. If you cannot determine the veracity, state that clearly."
                },
                {
                    "role": "user",
                    "content": f"Fact-check this: {text}"
                }
            ],
            model="gpt-4o",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during text fact-check: {e}")
        return f"Error during fact-check: {e}"

async def perform_image_factcheck(image_data: bytes, caption: str = None) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured for image fact-checking."

    # FIX 6: Prioritize imghdr for more robust MIME type detection
    mime_type = None
    img_type = imghdr.what(None, h=image_data)
    if img_type:
        mime_type = f"image/{img_type}"
    else:
        mime_type = "application/octet-stream" # Generic fallback

    image_parts = []
    image_parts.append({
        "mime_type": mime_type,
        "data": base64.b64encode(image_data).decode("utf-8")
    })

    model_name = "gemini-2.0-flash"
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    prompt_parts = [
        {"text": "Analyze the following image for any factual inaccuracies, misleading content, or provide context if it's being used deceptively. Be neutral and precise. If there's a caption, consider it in your analysis."},
        {"inline_data": image_parts[0]},
    ]
    if caption:
        prompt_parts.append({"text": f"Caption/Question: {caption}"})

    data = {
        "contents": [{"parts": prompt_parts}],
        "safety_settings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            response_data = response.json()
            if response_data and "candidates" in response_data and len(response_data["candidates"]) > 0:
                first_candidate = response_data["candidates"][0]
                if "content" in first_candidate and "parts" in first_candidate["content"]:
                    for part in first_candidate["content"]["parts"]:
                        if "text" in part:
                            return part["text"]
            return "No fact-check result could be extracted from the image."
    except httpx.HTTPStatusError as e:
        print(f"HTTP error during Gemini image fact-check: {e.response.status_code} - {e.response.text}")
        return f"Error processing image: {e.response.text}"
    except Exception as e:
        print(f"Error during Gemini image fact-check: {e}")
        return f"Error processing image: {e}"


# --- FastAPI Endpoints ---
@app.get("/")
async def read_root():
    # IMPORTANT CHANGE HERE: Serve index.html from the 'static' directory
    return FileResponse('static/index.html')

@app.post("/factcheck/audio")
async def factcheck_audio(audio_file: UploadFile = File(...)):
    transcription = await transcribe_audio(audio_file)
    if transcription.startswith("Error"):
        return JSONResponse(status_code=500, content={"error": transcription})

    fact_check_result = await perform_text_factcheck(transcription)

    audio_response = await text_to_speech(fact_check_result)
    if not audio_response:
        return JSONResponse(status_code=500, content={"error": "Failed to synthesize audio response."})

    return JSONResponse(content={
        "transcription": transcription,
        "factCheckResult": fact_check_result,
        "audio": base64.b64encode(audio_response).decode('utf-8')
    })

@app.post("/factcheck/text")
async def factcheck_text(item: dict):
    text = item.get("text")
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text input is required."})

    fact_check_result = await perform_text_factcheck(text)

    audio_response = await text_to_speech(fact_check_result)
    if not audio_response:
        return JSONResponse(status_code=500, content={"error": "Failed to synthesize audio response."})

    return JSONResponse(content={
        "factCheckResult": fact_check_result,
        "audio": base64.b64encode(audio_response).decode('utf-8')
    })

@app.post("/factcheck/image")
async def factcheck_image(image: UploadFile = File(...), caption: str = Form(None)):
    image_data = await image.read()
    if not image_data:
        return JSONResponse(status_code=400, content={"error": "Image file is required."})

    fact_check_result = await perform_image_factcheck(image_data, caption)
    if fact_check_result.startswith("Error") or "No fact-check result could be extracted" in fact_check_result:
        return JSONResponse(status_code=500, content={"error": fact_check_result})

    audio_response = await text_to_speech(fact_check_result)
    if not audio_response:
        return JSONResponse(status_code=500, content={"error": "Failed to synthesize audio response."})

    return JSONResponse(content={
        "factCheckResult": fact_check_result,
        "audio": base64.b64encode(audio_response).decode('utf-8')
    })

@app.post("/news/add_source")
async def add_news_source(source_url: str = Form(...), source_name: str = Form(...)):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO sources (name, url) VALUES (%s, %s) RETURNING id;", (source_name, source_url))
        source_id = cur.fetchone()[0]
        conn.commit()
        return JSONResponse(status_code=200, content={"message": "Source added successfully", "id": source_id, "name": source_name, "url": source_url})
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return JSONResponse(status_code=409, content={"error": "Source with this name or URL already exists."})
    except Exception as e:
        conn.rollback()
        print(f"Error adding news source: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to add source: {e}"})
    finally:
        if conn:
            conn.close()

@app.post("/news/remove_source")
async def remove_news_source(source_id: int = Form(...)):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM sources WHERE id = %s;", (source_id,))
        conn.commit()
        if cur.rowcount == 0:
            return JSONResponse(status_code=404, content={"error": "Source not found."})
        return JSONResponse(status_code=200, content={"message": "Source and its articles deleted successfully."})
    except Exception as e:
        conn.rollback()
        print(f"Error removing news source: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to remove source: {e}"})
    finally:
        if conn:
            conn.close()

@app.get("/news/sources")
async def get_news_sources():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT id, name, url FROM sources ORDER BY created_at DESC;")
        sources = cur.fetchall()
        return JSONResponse(status_code=200, content={"sources": [dict(s) for s in sources]})
    except Exception as e:
        print(f"Error fetching news sources: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch sources: {e}"})
    finally:
        if conn:
            conn.close()

@app.get("/news/articles")
async def get_news_articles(
    source_id: int = Query(None, description="Optional: Filter articles by source ID"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(10, ge=1, le=100, description="Limit for pagination")
):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        query = """
            SELECT a.title, a.link, a.pub_date, s.name AS source_name, s.url AS source_url
            FROM articles a
            JOIN sources s ON a.source_id = s.id
        """
        params = []

        if source_id is not None:
            query += " WHERE a.source_id = %s"
            params.append(source_id)

        query += " ORDER BY a.pub_date DESC NULLS LAST, a.fetched_at DESC LIMIT %s OFFSET %s;"
        params.extend([limit, offset])

        cur.execute(query, params)
        articles = cur.fetchall()

        # Check if there are more articles for pagination
        has_more = False
        count_query = "SELECT COUNT(*) FROM articles"
        count_params = []
        if source_id is not None:
            count_query += " WHERE source_id = %s"
            count_params.append(source_id)
        cur.execute(count_query, count_params)
        total_articles = cur.fetchone()[0]

        if total_articles > (offset + len(articles)):
            has_more = True

        all_articles = []
        for article in articles:
            # Ensure pub_date is a string in ISO format for consistent JSON serialization
            pub_date_str = article['pub_date'].isoformat() if article['pub_date'] else None
            all_articles.append({
                "title": article['title'],
                "link": article['link'],
                "source": article['source_name'],
                "pubDate": pub_date_str
            })

        return JSONResponse(content={"articles": all_articles, "hasMore": has_more}, status_code=200)

    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch news articles: {e}"})
    finally:
        if conn:
            conn.close()


@app.post("/news/fetch_and_store")
async def fetch_and_store_news(source_id: int):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cur.execute("SELECT id, url, name FROM sources WHERE id = %s;", (source_id,))
        source = cur.fetchone()
        if not source:
            return JSONResponse(status_code=404, content={"error": "Source not found."})

        feed = feedparser.parse(source['url'])
        if feed.bozo:
            if hasattr(feed.bozo_exception, 'status'):
                return JSONResponse(status_code=feed.bozo_exception.status, content={"error": f"Error fetching RSS feed: HTTP {feed.bozo_exception.status}"})
            else:
                return JSONResponse(status_code=500, content={"error": f"Error parsing RSS feed: {feed.bozo_exception}"})

        new_articles_count = 0
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            pub_date = None
            if hasattr(entry, 'published_parsed'):
                pub_date = datetime(*entry.published_parsed[:6])
            
            try:
                cur.execute("INSERT INTO articles (source_id, title, link, pub_date) VALUES (%s, %s, %s, %s) ON CONFLICT (link) DO NOTHING;",
                            (source['id'], title, link, pub_date))
                if cur.rowcount > 0:
                    new_articles_count += 1
            except Exception as e:
                print(f"Error inserting article {link}: {e}")
                pass

        conn.commit()
        cur.execute("UPDATE sources SET last_fetched = CURRENT_TIMESTAMP WHERE id = %s;", (source_id,))
        conn.commit()

        return JSONResponse(status_code=200, content={"message": f"Fetched and stored {new_articles_count} new articles for {source['name']}."})

    except Exception as e:
        if conn:
            conn.rollback() 
        print(f"Error in fetch_and_store_news for source_id {source_id}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch and store news: {e}"})
    finally:
        if conn:
            conn.close()


# --- Telegram Bot Webhook (Optional, for Render deployment) ---
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    if not TELEGRAM_BOT_TOKEN or not telegram_application:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content="Telegram bot not configured or initialized.")
    try:
        req_json = await request.json()
        update = Update.de_json(req_json, telegram_application.bot)
        
        if update: 
            await telegram_application.process_update(update)
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        print(f"Error processing Telegram webhook: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))

    if TELEGRAM_BOT_TOKEN and not RENDER_SERVICE_URL:
        if telegram_application:
            print("Running Telegram bot in local polling mode...")
            telegram_application.run_polling(poll_interval=1.0)
        else:
            print("Telegram bot token set, but initialization failed. Cannot run polling.")
    
    uvicorn.run(app, host="0.0.0.0", port=port)