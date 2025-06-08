import os
import io
import base64
import tempfile
import mimetypes # For MIME type inference from file extension
import imghdr # For robust image type inference from content
from pydub import AudioSegment # For audio processing (used in Telegram bot)
from google.cloud import texttospeech
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI # Keep OpenAI import for text/audio
from dotenv import load_dotenv
import json
import httpx # Import httpx for making asynchronous HTTP requests to Gemini

# Import Telegram bot setup from our new module (relative import)
from . import telegram_bot_handlers 
from telegram import Update # Import Update class directly here for de_json for webhook

load_dotenv()

# --- IMPORTANT ---
# Ensure your OPENAI_API_KEY is set in your .env file locally,
# and as an environment variable on Render.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Keep OpenAI client for text/audio processing AND for Telegram bot's text/audio transcription
openai_client = OpenAI(api_key=OPENAI_API_KEY) 

# --- Gemini API Config ---
# For Canvas runtime, GEMINI_API_KEY can be an empty string, as Canvas injects it.
# For Render deployment, it MUST be read from environment variables.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Read from environment variable
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

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

# --- FACT-CHECKING FUNCTIONS ---

# Function for Text Fact-Checking (uses OpenAI)
async def perform_text_factcheck(input_text: str):
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
"{input_text}"

Instructions:
- Respond in clear, simple Telugu — use English only where needed.
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

    ai_response = openai_client.chat.completions.create( # Using openai_client
        model="o4-mini",
        messages=messages
    )

    fact_check_result = ai_response.choices[0].message.content
    audio_base64 = text_to_speech(fact_check_result)
    return {"result": fact_check_result, "audio_result": audio_base64}

# Function for Audio Fact-Checking (uses OpenAI Whisper for transcription, then OpenAI chat for fact-check)
async def perform_audio_factcheck(audio_file_path: str):
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create( # Using openai_client
            model="whisper-1",
            file=audio_file
        )
        transcribed_text = transcript.text

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
"{transcribed_text}" # Using transcribed_text here

Instructions:
- Respond in clear, simple Telugu — use English only where needed.
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

    ai_response = openai_client.chat.completions.create( # Using openai_client
        model="o4-mini",
        messages=messages
    )

    fact_check_result = ai_response.choices[0].message.content
    audio_base64_result = text_to_speech(fact_check_result)
    return {"transcription": transcribed_text, "result": fact_check_result, "audio_result": audio_base64_result}

# NEW Unified Function for Image Fact-Checking using Gemini (for Website AND Telegram)
# Renamed from perform_image_factcheck_gemini to reflect its universal use.
async def perform_image_factcheck(image_bytes: bytes, mime_type: str, caption: str):
    # This function will be used by BOTH the website and the Telegram bot for image uploads
    
    # --- Robust Image Type Inference ---
    guessed_type = imghdr.what(None, h=image_bytes)
    
    mime_type_for_gemini = None 
    if guessed_type == 'jpeg':
        mime_type_for_gemini = 'image/jpeg'
    elif guessed_type == 'png':
        mime_type_for_gemini = 'image/png'
    elif guessed_type == 'gif':
        mime_type_for_gemini = 'image/gif'
    elif mime_type == 'image/webp': # WebP needs explicit check if imghdr doesn't catch
        mime_type_for_gemini = 'image/webp'
    
    if not mime_type_for_gemini:
        allowed_mime_types = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']
        if mime_type not in allowed_mime_types:
            raise ValueError(f"Unsupported image MIME type received: {mime_type}. Allowed: {allowed_mime_types}")
        mime_type_for_gemini = mime_type

    print(f"DEBUG: Original MIME type from FastAPI (Image Fact-Check): {mime_type}")
    print(f"DEBUG: Inferred MIME type for Gemini (Image Fact-Check): {mime_type_for_gemini}")
    print(f"DEBUG: Image bytes length (Image Fact-Check): {len(image_bytes)}")
        
    if len(image_bytes) == 0:
        raise ValueError("Empty image data received")
    
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    print(f"DEBUG: Base64 image length (Image Fact-Check): {len(b64_image)}")
    print(f"DEBUG: Base64 preview (first 50 chars - Image Fact-Check): {b64_image[:50]}...")
    # --- End Image Type Inference ---

    # --- GEMINI API CALL ---
    gemini_prompt_parts = [
        {
            "text": f"""You're given an image to fact-check. Your job is to analyze it like a knowledgeable, honest human — not like an AI.
            
            {f"Context provided: {caption}" if caption else "No additional context provided."}
            
            Instructions:
            - Respond in clear, simple Telugu — use English only where needed.
            - Do not respond in bullet points. Write like you're explaining it to someone directly.
            - Analyze what you see in the image and determine if it appears authentic or manipulated.
            - If you detect signs of manipulation, explain what you notice.
            - If it appears genuine, provide context about what the image shows.
            - If it's controversial or you're uncertain, be honest about limitations.
            - Do not repeat yourself.
            - Use natural sentence flow like a real person would.
            """
        },
        {
            "inlineData": {
                "mimeType": mime_type_for_gemini,
                "data": b64_image
            }
        }
    ]

    gemini_payload = {
        "contents": [
            {
                "role": "user",
                "parts": gemini_prompt_parts
            }
        ]
        # REMOVED: "tools" section to disable Google Search grounding
    }

    try:
        print("DEBUG: Sending request to Gemini API for image processing...")
        print(f"DEBUG: Using Gemini model 'gemini-2.0-flash' for image processing (no grounding).")
        
        async with httpx.AsyncClient() as http_client:
            gemini_response = await http_client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                json=gemini_payload,
                headers={"Content-Type": "application/json"}
            )
            gemini_response.raise_for_status() 
        
        gemini_result = gemini_response.json()
        
        fact_check_result = "No result text found."
        citations = [] # Citations will no longer be available without grounding
        if gemini_result.get('candidates') and len(gemini_result['candidates']) > 0 and \
           gemini_result['candidates'][0].get('content') and \
           gemini_result['candidates'][0]['content'].get('parts') and \
           len(gemini_result['candidates'][0]['content']['parts']) > 0:
            fact_check_result = gemini_result['candidates'][0]['content']['parts'][0].get('text', 'No result text found.')
            
            # Removed citation extraction as grounding is disabled
        else:
            print(f"DEBUG: Unexpected Gemini response structure for image: {gemini_result}")

        print("DEBUG: Gemini request successful (Image Fact-Check)")
        
        # Generate TTS for the Gemini fact-check result
        audio_base64 = text_to_speech(fact_check_result)

        return {"result": fact_check_result, "audio_result": audio_base64, "sources": citations}
    except httpx.HTTPStatusError as e:
        error_message = f"Gemini API HTTP Error (Image Fact-Check): Status {e.response.status_code}, Response: {e.response.text}"
        print(f"Error: {error_message}") 
        raise ValueError(error_message) 
    except json.JSONDecodeError as e:
        error_message = f"Failed to parse Gemini API response as JSON: {e}"
        print(f"Error: {error_message}")
        raise ValueError(error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred during Gemini image processing: {type(e).__name__}: {e}"
        print(f"Error: {error_message}", exc_info=True)
        raise ValueError(error_message)


# --- WEB APP ENDPOINTS ---
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
    """
    Fact-checks an image with an optional caption using Gemini's vision model for the website.
    Includes robust file type and header validation.
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": f"Invalid file type. Expected image, got: {file.content_type}"}, 
                status_code=400
            )
        
        print(f"DEBUG: Received file from website - Name: {file.filename}, Content-Type: {file.content_type}, Size: {file.size}")
        
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            return JSONResponse(
                content={"error": "Empty file received"}, 
                status_code=400
            )
            
        print(f"DEBUG: Read {len(image_bytes)} bytes from uploaded file (Website)")
        
        if len(image_bytes) < 8: 
            return JSONResponse(
                content={"error": "File too small to be a valid image"}, 
                status_code=400
            )
        
        header = image_bytes[:12] 
        is_valid_image = False
        detected_format = "Unknown"

        if header.startswith(b'\x89PNG\r\n\x1a\n'):  
            is_valid_image = True
            detected_format = "PNG"
        elif header.startswith(b'\xff\xd8\xff'):  
            is_valid_image = True
            detected_format = "JPEG"
        elif header.startswith(b'GIF8'):  
            is_valid_image = True
            detected_format = "GIF"
        elif header.startswith(b'RIFF') and len(image_bytes) >= 12 and image_bytes[8:12] == b'WEBP': 
            is_valid_image = True
            detected_format = "WebP"
                
        if not is_valid_image:
            return JSONResponse(
                content={"error": "File does not appear to be a valid image format"}, 
                status_code=400
            )
            
        print(f"DEBUG: Detected image format via magic number (Website): {detected_format}")
        
        # Call the unified Gemini-specific image processing function
        response = await perform_image_factcheck(image_bytes, file.content_type, caption)
        return JSONResponse(content=response)
            
    except ValueError as ve:
        print(f"Validation error in factcheck_image_web: {ve}")
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        print(f"Error in web factcheck_image: {e}")
        return JSONResponse(content={"error": f"Internal server error: {str(e)}"}, status_code=500)

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
    # Pass the unified Gemini image fact-check function for the Telegram bot
    telegram_bot_handlers.initialize_bot_components(
        openai_client_instance=openai_client, 
        tts_func=text_to_speech,
        authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
        perform_image_factcheck_func=perform_image_factcheck # <--- IMPORTANT: Now using unified Gemini function
    )

    if TELEGRAM_BOT_TOKEN:
        telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
        await telegram_application.initialize()
        
        webhook_base_url = os.getenv("RENDER_SERVICE_URL")
        
        if webhook_base_url:
            full_webhook_url = f"{webhook_base_url}/telegram-webhook"
            print(f"Setting Telegram webhook to: {full_webhook_url}")
            try:
                await telegram_application.bot.set_webhook(url=full_webhook_url)
                print("Telegram webhook set successfully.")
            except Exception as webhook_e:
                print(f"Error setting Telegram webhook: {webhook_e}")
        else:
            print("RENDER_SERVICE_URL not found. Webhook not set for production. Running in local polling mode (if __name__ is called).")
    else:
        print("TELEGRAM_BOT_TOKEN not found. Telegram bot functionality will not be active.")

@app.on_event("shutdown")
async def shutdown_event():
    """Remove the webhook for the Telegram bot when the FastAPI app shuts down."""
    if telegram_application and os.getenv("RENDER_SERVICE_URL"): 
        print("Removing Telegram webhook...")
        await telegram_application.shutdown()
        print("Telegram webhook removed (via application shutdown).")

# --- TELEGRAM WEBHOOK ENDPOINT ---
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request) -> Response:
    """Handle incoming Telegram updates."""
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

# --- Main execution block ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    if TELEGRAM_BOT_TOKEN and not os.getenv("RENDER_SERVICE_URL"): 
        print("Running Telegram bot in local polling mode...")
        # from telegram import Update # Not needed here, only for webhook
        telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
        telegram_bot_handlers.initialize_bot_components(
            openai_client_instance=openai_client, 
            tts_func=text_to_speech,
            authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
            perform_image_factcheck_func=perform_image_factcheck # <--- IMPORTANT: Now using unified Gemini function
        )
        telegram_application.run_polling(poll_interval=1)
    else:
        print("Running Uvicorn server for web app and webhook endpoint...")
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
