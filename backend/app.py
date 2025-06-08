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
from openai import OpenAI
from dotenv import load_dotenv
import json

# Import Telegram bot setup from our new module (relative import)
from . import telegram_bot_handlers 

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

    ai_response = client.chat.completions.create(
        model="o4-mini",
        # temperature=0.2, # REMOVED: o4-mini does not support custom temperature
        messages=messages
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

    ai_response = client.chat.completions.create(
        model="o4-mini",
        # temperature=0.2, # REMOVED: o4-mini does not support custom temperature
        messages=messages
    )

    fact_check_result = ai_response.choices[0].message.content
    audio_base64_result = text_to_speech(fact_check_result)
    return {"transcription": transcribed_text, "result": fact_check_result, "audio_result": audio_base64_result}

async def perform_image_factcheck(image_bytes: bytes, mime_type: str, caption: str):
    # --- START NEW IMAGE TYPE INFERENCE AND VALIDATION LOGIC ---
    # Try to guess mime type from image content, which is more robust than just header
    # and map it to common types OpenAI expects.
    guessed_type = imghdr.what(None, h=image_bytes)
    
    mime_type_for_openai = None
    if guessed_type == 'jpeg':
        mime_type_for_openai = 'image/jpeg'
    elif guessed_type == 'png':
        mime_type_for_openai = 'image/png'
    elif guessed_type == 'gif':
        mime_type_for_openai = 'image/gif'
    # imghdr does not support webp directly, so we trust the original mime_type if it's webp
    elif mime_type == 'image/webp': # Check if original MIME type is webp
        mime_type_for_openai = 'image/webp'
    
    # Fallback to original mime_type if imghdr couldn't guess or it's not a common type
    if not mime_type_for_openai:
        # Validate against OpenAI's explicitly supported list
        allowed_mime_types = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']
        if mime_type not in allowed_mime_types:
            raise ValueError(f"Unsupported image MIME type received: {mime_type}. Allowed: {allowed_mime_types}")
        mime_type_for_openai = mime_type # Trust the original if it's in the allowed list

    print(f"DEBUG: Original MIME type from FastAPI: {mime_type}")
    print(f"DEBUG: Inferred MIME type for OpenAI: {mime_type_for_openai}")
    print(f"DEBUG: Image bytes length: {len(image_bytes)}")
        
    # Validate image bytes are not empty
    if len(image_bytes) == 0:
        raise ValueError("Empty image data received")
    
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    print(f"DEBUG: Base64 image length: {len(b64_image)}")
    print(f"DEBUG: Base64 preview (first 50 chars): {b64_image[:50]}...")
        
    # Create the data URL - this is the critical part
    data_url = f"data:{mime_type_for_openai};base64,{b64_image}"
    # --- END NEW IMAGE TYPE INFERENCE AND VALIDATION LOGIC ---

    messages = [
        {
            "role": "system",
            "content": "You are a smart and honest Telugu-speaking fact checker. You speak like a well-informed, friendly human who mixes Telugu and English naturally. You never repeat yourself. Be accurate, clear, and real."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Is this a real photo of a unicorn?"},
                {"type": "image_url", "image_url": {"url": "https://placehold.co/200x200/FF0000/FFFFFF?text=Unicorn"}} # Dummy image for example
            ]
        },
        {
            "role": "assistant",
            "content": "ఇది నిజమైన యునికార్న్ ఫోటో కాదు. యునికార్న్‌లు పురాణ జీవులు, అవి నిజంగా ఉనికిలో లేవు. ఇది డిజిటల్‌గా సృష్టించబడిన చిత్రం లేదా నకిలీ ఫోటో అయ్యే అవకాశం ఉంది."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": f"""You're given an image to fact-check. Your job is to analyze it like a knowledgeable, honest human — not like an AI.{f"Context provided: {caption}" if caption else "No additional context provided."}Instructions:- Respond in clear, simple Telugu — use English only where needed.- Do not respond in bullet points. Write like you're explaining it to someone directly.- Analyze what you see in the image and determine if it appears authentic or manipulated.- If you detect signs of manipulation, explain what you notice.- If it appears genuine, provide context about what the image shows.- If it's controversial or you're uncertain, be honest about limitations.- Do not repeat yourself.- Use natural sentence flow like a real person would."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url} # Use the dynamically created data_url
                }
            ],
        }
    ]

    try:
        print("DEBUG: Sending request to OpenAI...")
        model_name = "gpt-4o" # Explicitly state the model used for debugging
        print(f"DEBUG: Using model '{model_name}' for image processing.")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500
        )
        print("DEBUG: OpenAI request successful")
        return {"result": response.choices[0].message.content}
    except Exception as e:
        print(f"Error calling OpenAI API for image: {e}")
        # Additional debugging for OpenAI errors
        if hasattr(e, 'response') and e.response:
            print(f"OpenAI API response status: {e.response.status_code}")
            print(f"OpenAI API response body: {e.response.text}")
        # Re-raise the exception after printing for visibility in logs
        raise # Re-raise to be caught by the FastAPI endpoint handler

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
    Fact-checks an image with an optional caption using OpenAI's vision model.
    Includes robust file type and header validation.
    """
    try:
        # Validate file is actually an image based on content_type header
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": f"Invalid file type. Expected image, got: {file.content_type}"}, 
                status_code=400
            )
        
        print(f"DEBUG: Received file - Name: {file.filename}, Content-Type: {file.content_type}, Size: {file.size}")
        
        image_bytes = await file.read()
        
        # Validate we actually got image data
        if len(image_bytes) == 0:
            return JSONResponse(
                content={"error": "Empty file received"}, 
                status_code=400
            )
            
        print(f"DEBUG: Read {len(image_bytes)} bytes from uploaded file")
        
        # Additional validation - check if the image bytes start with valid image headers (magic numbers)
        if len(image_bytes) < 8: # Minimum size for most common image headers
            return JSONResponse(
                content={"error": "File too small to be a valid image"}, 
                status_code=400
            )
        
        header = image_bytes[:12] # Check first few bytes
        is_valid_image = False
        detected_format = "Unknown"

        if header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
            is_valid_image = True
            detected_format = "PNG"
        elif header.startswith(b'\xff\xd8\xff'):  # JPEG
            is_valid_image = True
            detected_format = "JPEG"
        elif header.startswith(b'GIF8'):  # GIF
            is_valid_image = True
            detected_format = "GIF"
        # WebP magic number: RIFF (52 49 46 46) at start, and WEBP at offset 8
        elif header.startswith(b'RIFF') and len(image_bytes) >= 12 and image_bytes[8:12] == b'WEBP': 
            is_valid_image = True
            detected_format = "WebP"
                
        if not is_valid_image:
            return JSONResponse(
                content={"error": "File does not appear to be a valid image format"}, 
                status_code=400
            )
            
        print(f"DEBUG: Detected image format via magic number: {detected_format}")
        
        # Pass file.content_type to perform_image_factcheck, which will then use imghdr
        response = await perform_image_factcheck(image_bytes, file.content_type, caption)
        return JSONResponse(content=response)
            
    except ValueError as ve: # Catch validation errors raised by perform_image_factcheck
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
    telegram_bot_handlers.initialize_bot_components(
        openai_client_instance=client,
        tts_func=text_to_speech,
        authorized_ids=AUTHORIZED_TELEGRAM_USER_IDS,
        perform_image_factcheck_func=perform_image_factcheck
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
        # Ensure Update is imported from telegram for de_json
        from telegram import Update 
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
        # Make sure to import Update if running this block directly for telegram.Update.de_json
        from telegram import Update 
        telegram_application = telegram_bot_handlers.setup_telegram_bot_application(TELEGRAM_BOT_TOKEN)
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

