from google.cloud import texttospeech
from fastapi import FastAPI, UploadFile, File, Form
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
import json # Import json module

load_dotenv()

# --- IMPORTANT ---
# Ensure your OPENAI_API_KEY is set in your .env file locally,
# and as an environment variable on Render.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Serve static frontend files
# This assumes your 'frontend' directory is a sibling to your 'backend' directory
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

# --- MODIFICATION FOR GOOGLE TTS CREDENTIALS ON RENDER ---
# On Render, you will set the actual JSON content of your gcloud_key.json
# file as an environment variable named GOOGLE_CREDENTIALS_JSON.
# This code reads that variable and writes it to a temporary file,
# then sets GOOGLE_APPLICATION_CREDENTIALS to point to this temp file.
google_credentials_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON")

if google_credentials_json_str:
    try:
        # Create a temporary file to store the credentials
        temp_gcloud_key_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_gcloud_key_file.write(google_credentials_json_str.encode('utf-8'))
        temp_gcloud_key_file.close() # Ensure file is closed to flush content

        # Set the environment variable to point to the temporary file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_gcloud_key_file.name
        print(f"Google Cloud credentials loaded from environment variable to: {temp_gcloud_key_file.name}")
    except Exception as e:
        print(f"Error processing GOOGLE_CREDENTIALS_JSON environment variable: {e}")
        # If there's an error, TTS might fail, but the app can still try to run.
else:
    print("GOOGLE_CREDENTIALS_JSON environment variable not found. Google TTS may not work.")
    # Fallback for local development if you prefer to set it manually for testing:
    # Example for local development if gcloud_key.json is in 'backend' directory
    # If you choose this, make sure to uncomment and provide the correct path.
    # try:
    #     local_key_path = os.path.join(os.path.dirname(__file__), "gcloud_key.json")
    #     if os.path.exists(local_key_path):
    #         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_key_path
    #         print(f"Using local Google Cloud credentials from: {local_key_path}")
    # except Exception as e:
    #     print(f"Error setting local GOOGLE_APPLICATION_CREDENTIALS: {e}")


# Initialize Google Cloud TTS client
tts_client = texttospeech.TextToSpeechClient()

def text_to_speech(text: str) -> str:
    """
    Converts text to speech using Google Cloud TTS and returns base64 encoded MP3 audio.
    Assumes Telugu language and a specific female voice.
    """
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="te-IN",  # Telugu (India)
        name="te-IN-Chirp3-HD-Achird", # Specific voice for Telugu
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Return base64 encoded audio content
    return base64.b64encode(response.audio_content).decode('utf-8')

@app.post("/factcheck/text")
async def factcheck_text(input: TextInput):
    """
    Fact-checks the provided text using OpenAI and generates a TTS audio response
    for the fact-check result.
    """
    prompt = f"Fact-check this text and provide a clear answer:\n\n{input.text}"
    try:
        # Get fact-check result from OpenAI
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You're a helpful fact-checking assistant. Respond clearly and concisely."},
                {"role": "user", "content": prompt}
            ]
        )
        fact_check_result = ai_response.choices[0].message.content

        # Generate TTS for the fact-check result
        audio_base64 = text_to_speech(fact_check_result)

        return {
            "result": fact_check_result,
            "audio_result": audio_base64 # Include the audio of the result
        }
    except Exception as e:
        print(f"Error in factcheck_text: {e}")
        return {"error": str(e)}

@app.post("/factcheck/audio")
async def factcheck_audio(file: UploadFile = File(...)):
    """
    Transcribes uploaded audio using OpenAI Whisper, fact-checks the transcription,
    and generates a TTS audio response for the fact-check result.
    """
    # Create a temporary file to save the uploaded audio
    # Using .webm suffix as it's common for recorded audio from browsers
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Transcribe audio using OpenAI Whisper
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            transcribed_text = transcript.text

        # Fact-check the transcribed text using OpenAI
        prompt = f"Fact-check this audio content (transcribed in Telugu):\n\n{transcribed_text}"
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You're a Telugu fact-checking assistant. Provide clear and concise answers in Telugu."},
                {"role": "user", "content": prompt}
            ]
        )
        fact_check_result = ai_response.choices[0].message.content

        # Generate TTS for the fact-check result
        audio_base64_result = text_to_speech(fact_check_result)

        return {
            "transcription": transcribed_text,
            "result": fact_check_result,
            "audio_result": audio_base64_result # Include the audio of the result
        }
    except Exception as e:
        print(f"Error in factcheck_audio: {e}")
        return {"error": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/factcheck/image")
async def factcheck_image(
    file: UploadFile = File(...),
    caption: str = Form("")
):
    """
    Fact-checks an image with an optional caption using OpenAI's vision model.
    No TTS generation for image results in this endpoint, as per original request.
    """
    try:
        # Read image bytes and base64 encode for OpenAI Vision API
        image_bytes = await file.read()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Construct messages with embedded image in data URI format
        messages = [
            {"role": "system", "content": "You are a fact-checking assistant. Provide clear and concise answers."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption or "Please fact-check this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{file.content_type};base64,{b64_image}"
                        }
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

    except Exception as e:
        print(f"Error in factcheck_image: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/factcheck/tts")
async def generate_tts(input: TextInput):
    """
    Generates TTS audio from provided text. This endpoint is kept for direct TTS use
    but is not directly used for speaking fact-check results by the modified frontend.
    """
    try:
        audio_base64 = text_to_speech(input.text)
        return {"audio": audio_base64}
    except Exception as e:
        print(f"Error in generate_tts: {e}")
        return {"error": str(e)}

# Main execution block
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use PORT env var or default 8000 for local testing
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

