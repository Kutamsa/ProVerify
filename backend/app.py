import os
import io
import base64
import tempfile
import imghdr
from pydub import AudioSegment
from google.cloud import texttospeech
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json
import httpx
from contextlib import asynccontextmanager

from . import telegram_bot_handlers
from telegram import Update
from telegram.ext import Application # Import Application directly for initialize()

import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone # Import timezone for robust datetime handling
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

# Global variables, initialized in lifespan
tts_client = None
telegram_application = None

# Initialize OpenAI Client globally as it doesn't depend on lifespan
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Database Connection ---
def get_db_connection():
    if not DATABASE_URL:
        print("DATABASE_URL environment variable is not set.")
        return None
    try:
        result = urlparse(DATABASE_URL)
        username = result.username
        password = result.password
        database = result.path[1:]
        hostname = result.hostname
        port = result.port
        conn = psycopg2.connect(
            database=database,
            user=username,
            password=password,
            host=hostname,
            port=port
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# --- Database Initialization (PostgreSQL) ---
def init_news_db():
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("Skipping DB initialization due to connection error.")
            return

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

        # Add ALTER TABLE statement to add 'last_fetched' column if it doesn't exist
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='sources' AND column_name='last_fetched') THEN
                    ALTER TABLE sources ADD COLUMN last_fetched TIMESTAMP WITH TIME ZONE;
                END IF;
            END
            $$;
        """)
        conn.commit()

        print("Database tables 'sources' and 'articles' ensured to exist, and 'last_fetched' column checked/added.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()


# --- Lifespan Context for FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_client, telegram_application

    # Initialize Google TTS client and credentials
    if GOOGLE_CREDENTIALS_JSON_CONTENT:
        try:
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            temp_file.write(GOOGLE_CREDENTIALS_JSON_CONTENT)
            temp_file.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name # Set env var for Google client
            tts_client = texttospeech.TextToSpeechClient()
            print("Google Text-to-Speech client initialized.")
        except Exception as e:
            print(f"Error initializing Google TTS client: {e}")
            tts_client = None
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
    else:
        print("GOOGLE_CREDENTIALS_JSON environment variable not set. Google TTS will be unavailable.")

    # Initialize database
    init_news_db()

    # Initialize Telegram Bot components ONLY within lifespan
    if TELEGRAM_BOT_TOKEN:
        try:
            telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
            await telegram_application.initialize() 
            print("Telegram Application initialized inside lifespan.")

            telegram_bot_handlers.initialize_bot_components(
                openai_client_instance=openai_client,
                tts_func=text_to_speech if tts_client else None, # Pass actual TTS func if client is ready
                authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
                perform_image_factcheck_func=perform_image_factcheck, # Pass the unified image fact-check function
                transcribe_audio_func=transcribe_audio_from_bytes # Pass the new transcribe_audio_from_bytes function
            )
            print("Telegram bot components initialized within lifespan.")
        except Exception as e:
            print(f"Error initializing Telegram bot components in lifespan: {e}")
            telegram_application = None
    else:
        print("TELEGRAM_BOT_TOKEN not set. Telegram bot will not be active.")

    yield # Application starts here

    # Cleanup on shutdown
    if GOOGLE_CREDENTIALS_JSON_CONTENT and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        temp_file_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            print(f"Cleaned up temporary Google credentials file: {temp_file_path}")


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
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "..", "frontend") 
app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")


# --- Utility Functions ---

class TextInput(BaseModel):
    text: str

def text_to_speech(text: str) -> str: # Modified to return base64 string directly
    """Converts text to speech using Google Cloud TTS and returns base64 encoded MP3 audio."""
    if not tts_client:
        print("TTS client not available for synthesis.")
        return "" # Return empty string if client not available
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
        return base64.b64encode(response.audio_content).decode('utf-8')
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return ""

async def transcribe_audio_from_bytes(audio_bytes: bytes) -> str: # NEW: Function accepts bytes directly
    """Transcribes audio content (bytes) using OpenAI Whisper API."""
    # Create an in-memory file-like object for OpenAI API
    audio_bytes_io = io.BytesIO(audio_bytes)
    audio_bytes_io.name = "audio.mp3" # Assign a name, though not strictly required by OpenAI
    
    try:
        # OpenAI Whisper API for transcription
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes_io # Pass the BytesIO object
        )
        return transcript.text
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return f"Error transcribing audio: {e}"
    finally:
        audio_bytes_io.close() # Close the in-memory BytesIO object

async def perform_text_factcheck(input_text: str):
    if not OPENAI_API_KEY:
        return {"error": "Error: OpenAI API key is not configured for text fact-checking."}
    
    messages = [
        {
            "role": "system",
            "content": "You are a smart and honest Telugu-speaking fact checker. You speak like a well-informed, friendly human who mixes Telugu and English naturally. You never repeat yourself. Be accurate, clear, and real."
        },
        {
            "role": "user",
            "content": "మోదీ అమెరికా ప్రదాని"
        },
        {
            "role": "assistant",
            "content": "అది తప్పు. మోదీ భారతదేశ ప్రధాని. అమెరికా అధ్యక్షుడు Donald trump. కొన్నిసార్లు ప్రజలు ఈ విషయాన్ని తప్పుగా వినవచ్చు లేదా ప్రచారం చేయవచ్చు, కానీ ఇది నిజం కాదు."
        },
        {
            "role": "user",
            "content": f"""
You're given a statement. Your job is to fact-check it and respond like a knowledgeable, honest human — not like an AI.

Statement:
"{input_text}"

Instructions:
- Respond in clear, simple Telugu using its script - use english words only when they cannot be avoided.
- Do not respond in bullet points. Write like you're explaining it to someone directly.
- If the statement is false, explain why.
- If it's true, provide some brief context or clarification.
- If it's controversial, be honest and neutral. Don't dodge the question.
- If you don't know, say so clearly.
- Do not repeat yourself.
- Use natural sentence flow like a real person would.
"""
        }
    ]

    try:
        ai_response = openai_client.chat.completions.create(
            model="o4-mini",
            messages=messages
        )
        fact_check_result = ai_response.choices[0].message.content
        audio_base64 = text_to_speech(fact_check_result)
        return {"result": fact_check_result, "audio_result": audio_base64}
    except Exception as e:
        print(f"Error during text fact-check: {e}")
        return {"error": f"Error during fact-check: {e}"}

async def perform_audio_factcheck(audio_file_content: bytes): # This function now directly receives bytes
    if not OPENAI_API_KEY:
        return {"error": "Error: OpenAI API key is not configured for audio fact-checking."}

    try:
        # Pass the audio_file_content (bytes) directly to the new transcribe_audio_from_bytes function
        transcribed_text = await transcribe_audio_from_bytes(audio_file_content) 
        if transcribed_text.startswith("Error"):
            return {"error": transcribed_text}

        messages = [
            {
                "role": "system",
                "content": "You are a smart and honest Telugu-speaking fact checker. You speak like a well-informed, friendly human who mixes Telugu and English naturally. You never repeat yourself. Be accurate, clear, and real."
            },
            {
                "role": "user",
                "content": "మోదీ అమెరికా ప్రదాని"
            },
            {
                "role": "assistant",
                "content": "అది తప్పు. మోదీ భారతదేశ ప్రధాని. అమెరికా అధ్యక్షుడు జో బైడెన్. కొన్నిసార్లు ప్రజలు ఈ విషయాన్ని తప్పుగా వినవచ్చు లేదా ప్రచారం చేయవచ్చు, కానీ ఇది నిజం కాదు."
            },
            {
                "role": "user",
                "content": f"""
You're given a statement in Telugu. Your job is to fact-check it and respond like a knowledgeable, honest human — not like an AI.

Statement:
"{transcribed_text}" 

Instructions:
- Respond in clear, simple Telugu using its script - use english words only when they cannot be avoided.
- Do not respond in bullet points. Write like you're explaining it to someone directly.
- If the statement is false, explain why.
- If it's true, provide some brief context or clarification.
- If it's controversial, be honest and neutral. Don't dodge the question.
- If you don't know, say so clearly.
- Do not repeat yourself.
- Use natural sentence flow like a real person would.
"""
            }
        ]

        ai_response = openai_client.chat.completions.create(
            model="o4-mini",
            messages=messages
        )

        fact_check_result = ai_response.choices[0].message.content
        audio_base64_result = text_to_speech(fact_check_result)
        return {"transcription": transcribed_text, "result": fact_check_result, "audio_result": audio_base64_result}
    except Exception as e:
        print(f"Error during audio fact-check: {e}")
        return {"error": f"Error during fact-check: {e}"}


async def perform_image_factcheck(image_bytes: bytes, mime_type: str, caption: str = None):
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key is not configured for image fact-checking."}

    # Robust MIME type detection
    img_type = imghdr.what(None, h=image_bytes)
    mime_type_for_gemini = None

    if img_type:
        mime_type_for_gemini = f"image/{img_type}"
    else:
        if mime_type == 'image/webp':
            mime_type_for_gemini = 'image/webp'
        else:
            print("Warning: imghdr could not determine image type. Falling back to octet-stream or original MIME type if allowed.")
            allowed_mime_types = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']
            if mime_type in allowed_mime_types:
                mime_type_for_gemini = mime_type
            else:
                return {"error": f"Unsupported image MIME type: {mime_type}. Allowed: {', '.join(allowed_mime_types)}"}

    if len(image_bytes) == 0:
        return {"error": "Empty image data received"}
    
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    model_name = "gemini-2.5-flash-preview-05-20"
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    prompt_parts = [
        {
            "text": f"""You are an expert fact-checker. You are given an image. Your job is to analyze the text or message in the image and determine, as accurately as possible, if it is factual or contains misinformation.
            
            {f"Context provided: {caption}" if caption else "No additional context provided."}
            
            Instructions:
            - Your answer must be concise and must not exceed 200 words.Summarize your reasoning if needed to stay within the word limit.
            - Respond in clear, natural Telugu script. Use English only for words that cannot be translated.
            - Do not use bullet points; write in a conversational, explanatory tone.
            - First, briefly describe what the image says or shows.
            - Next, fact-check the key claim(s) or message in the image using your knowledge and reasoning. If possible, explain your reasoning step by step.
            - If the image contains incorrect or misleading information, clearly explain why, and provide the correct facts.
            - If the claim is controversial or evidence is mixed, mention this and explain both sides neutrally.
            - If the image is unclear or lacks enough information to fact-check, state this and explain what is missing.
            - Do not repeat yourself. Write as if you are explaining to a friend.
            - Your answer must be concise and must not exceed 200 words.Summarize your reasoning if needed to stay within the word limit.
            """
        },
        {
            "inlineData": {
                "mimeType": mime_type_for_gemini,
                "data": b64_image
            }
        }
    ]

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
            
            fact_check_result = "No result text found."
            if response_data.get('candidates') and len(response_data['candidates']) > 0 and \
               response_data['candidates'][0].get('content') and \
               response_data['candidates'][0]['content'].get('parts') and \
               len(response_data['candidates'][0]['content']['parts']) > 0:
                fact_check_result = response_data['candidates'][0]['content']['parts'][0].get('text', 'No result text found.')
            else:
                print(f"DEBUG: Unexpected Gemini response structure for image: {response_data}")

            audio_base64 = text_to_speech(fact_check_result)
            return {"result": fact_check_result, "audio_result": audio_base64, "sources": []} # No citations without grounding
    except httpx.HTTPStatusError as e:
        error_message = f"Gemini API HTTP Error (Image Fact-Check): Status {e.response.status_code}, Response: {e.response.text}"
        print(f"Error: {error_message}")
        return {"error": error_message}
    except json.JSONDecodeError as e:
        error_message = f"Failed to parse Gemini API response as JSON: {e}"
        print(f"Error: {error_message}")
        return {"error": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred during Gemini image processing: {type(e).__name__}: {e}"
        print(f"Error: {error_message}", exc_info=True)
        return {"error": error_message}


# --- FastAPI Endpoints ---
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(frontend_dir, 'index.html'))

@app.post("/factcheck/audio")
async def factcheck_audio_web(audio_file: UploadFile = File(...)):
    # Read content directly into memory
    content = await audio_file.read()
    response = await perform_audio_factcheck(content) # Pass raw bytes to perform_audio_factcheck
    if "error" in response:
        return JSONResponse(status_code=500, content=response)
    return JSONResponse(content=response)

@app.post("/factcheck/text")
async def factcheck_text_web(item: TextInput):
    response = await perform_text_factcheck(item.text)
    if "error" in response:
        return JSONResponse(status_code=500, content=response)
    return JSONResponse(content=response)

@app.post("/factcheck/image")
async def factcheck_image_web(
    image: UploadFile = File(...),
    caption: str = Form(None)
):
    image_data = await image.read()
    response = await perform_image_factcheck(image_data, image.content_type, caption)
    if "error" in response:
        return JSONResponse(status_code=500, content=response)
    return JSONResponse(content=response)

@app.post("/news/add_source")
async def add_news_source(source_url: str = Form(...), source_name: str = Form(...)):
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return JSONResponse(status_code=500, content={"error": "Database connection not established."})
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
        if not conn:
            return JSONResponse(status_code=500, content={"error": "Database connection not established."})
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
        if not conn:
            return JSONResponse(status_code=500, content={"error": "Database connection not established."})
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT id, name, url FROM sources ORDER BY id DESC;") 
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
        if not conn:
            return JSONResponse(status_code=500, content={"error": "Database connection not established."})
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

        query += " ORDER BY a.fetched_at DESC NULLS LAST LIMIT %s OFFSET %s;"
        params.extend([limit, offset])

        cur.execute(query, params)
        articles = cur.fetchall()

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
            pub_date_obj = None
            if article['pub_date']:
                if isinstance(article['pub_date'], datetime):
                    pub_date_obj = article['pub_date']
                else: 
                    try:
                        pub_date_obj = datetime.fromisoformat(str(article['pub_date']))
                    except ValueError:
                        try: 
                            pub_date_obj = datetime.strptime(str(article['pub_date']), '%a, %d %b %Y %H:%M:%S %z')
                        except ValueError:
                            print(f"Warning: Could not parse pub_date string: {article['pub_date']}")
                            pub_date_obj = None 
            
            pub_date_str = pub_date_obj.isoformat() if pub_date_obj else None
            
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
        if not conn:
            return JSONResponse(status_code=500, content={"error": "Database connection not established."})
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
                pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            
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
            if not telegram_application.updater and not telegram_application.bot_data:
                print("Warning: Telegram Application might not have been fully initialized. Attempting to process update anyway.")
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
            import asyncio
            asyncio.run(telegram_application.initialize())
            print("Running Telegram bot in local polling mode...")
            telegram_application.run_polling(poll_interval=1.0)
        else:
            telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
            telegram_bot_handlers.initialize_bot_components(
                openai_client_instance=openai_client,
                tts_func=text_to_speech,
                authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
                perform_image_factcheck_func=perform_image_factcheck,
                transcribe_audio_func=transcribe_audio_from_bytes # Pass the updated function
            )
            import asyncio
            asyncio.run(telegram_application.initialize())
            print("Running Telegram bot in local polling mode...")
            telegram_application.run_polling(poll_interval=1.0)
    else:
        print("Running Uvicorn server for web app and webhook endpoint...")
        uvicorn.run(app, host="0.0.0.0", port=port)
