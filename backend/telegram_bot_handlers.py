import os
import io
import base64
import tempfile 
import mimetypes 
from pydub import AudioSegment
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode 

# These global variables will be set during the bot's initialization
# by the main app.py file.
_openai_client = None
_tts_function = None
_authorized_user_ids = []
_perform_image_factcheck_function = None
_transcribe_audio_function = None 

def initialize_bot_components(
    openai_client_instance,
    tts_func,
    authorized_ids,
    perform_image_factcheck_func,
    transcribe_audio_func 
):
    """
    Initializes the necessary components for the Telegram bot handlers.
    This function should be called once from app.py on startup.
    """
    global _openai_client, _tts_function, _authorized_user_ids, _perform_image_factcheck_function, _transcribe_audio_function
    _openai_client = openai_client_instance
    _tts_function = tts_func
    _authorized_user_ids = authorized_ids
    _perform_image_factcheck_function = perform_image_factcheck_func
    _transcribe_audio_function = transcribe_audio_func 
    print("Telegram bot components initialized.")

def is_authorized(user_id: int) -> bool:
    """Checks if a user is authorized to use the bot."""
    return user_id in _authorized_user_ids

async def _perform_text_factcheck(text: str) -> str:
    if not _openai_client:
        return "Error: OpenAI client not initialized for text fact-checking."
    try:
        # The prompt and model for fact-checking text from Telegram will now mirror app.py's implementation
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
"{text}"

Instructions:
- Respond in clear, simple Telugu using its native script (తెలుగు లిపిలో) — use English only where needed.
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
        chat_completion = _openai_client.chat.completions.create(
            model="o4-mini",
            messages=messages
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during text fact-check in bot: {e}")
        return f"An error occurred during text fact-check: {e}"

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("You are not authorized to use this bot.")
        print(f"Unauthorized access attempt by user ID: {user_id}")
        return

    welcome_message = (
        "Hello! I'm your Fact-Checking Bot. "
        "Send me a text message, voice message, or a photo with a caption, "
        "and I will try to fact-check it for you.\n\n"
        "To get help, type /help."
    )
    await update.message.reply_text(welcome_message)
    print(f"Start command received from authorized user: {user_id}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    help_message = (
        "Here's how you can use me:\n\n"
        "• Send a *text message*: I will provide a fact-check for the text.\n"
        "• Send a *voice message*: I will transcribe it and then fact-check the transcription.\n"
        "• Send a *photo* (with or without a caption): I will analyze the image (and caption) for factual accuracy.\n\n"
        "Remember, I strive to be neutral and precise. If I can't determine the veracity, I will let you know."
    )
    await update.message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN)
    print(f"Help command received from authorized user: {user_id}")

# Mock UploadFile class for transcribe_audio function compatibility
# This class needs to be defined here to be used by handle_message
class MockUploadFile:
    def __init__(self, filename: str, content_type: str, file_content: bytes):
        self._file_content = io.BytesIO(file_content)
        self.filename = filename
        self.content_type = content_type

    async def read(self) -> bytes:
        self._file_content.seek(0)
        return self._file_content.read()

    # Async context manager methods are needed for FastAPI's UploadFile compatibility
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._file_content.close()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    message = update.message
    chat_id = message.chat_id

    try:
        if message.text:
            text = message.text
            await message.reply_text("Fact-checking your text...", quote=True)
            result = await _perform_text_factcheck(text)
            await message.reply_text(result)
            
            if _tts_function:
                try:
                    audio_bytes_base64 = _tts_function(result) # This now returns base64 string
                    audio_bytes = base64.b64decode(audio_bytes_base64) # Decode back to bytes
                    audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                    ogg_output = io.BytesIO()
                    audio.export(ogg_output, format="ogg", codec="libopus")
                    ogg_output.seek(0)
                    await context.bot.send_voice(chat_id=chat_id, voice=ogg_output, caption="Here's the audio fact-check:")
                except Exception as audio_e:
                    print(f"Error sending audio for text fact-check: {audio_e}")
                    await message.reply_text("Could not send audio response for the fact-check.")

        elif message.voice and message.voice.file_id:
            if not _transcribe_audio_function:
                await message.reply_text("Voice transcription service is not available.")
                return

            await message.reply_text("Processing voice message for transcription...", quote=True)
            voice_file = await context.bot.get_file(message.voice.file_id)
            
            voice_ogg_bytes_io = io.BytesIO()
            try:
                await voice_file.download_to_memory(voice_ogg_bytes_io)
                voice_ogg_bytes_io.seek(0)

                audio = AudioSegment.from_ogg(voice_ogg_bytes_io)
                mp3_output_io = io.BytesIO()
                audio.export(mp3_output_io, format="mp3")
                mp3_output_io.seek(0)

                # Create a MockUploadFile from the in-memory MP3 content
                # This ensures compatibility with app.py's transcribe_audio function
                mock_upload_file = MockUploadFile(
                    filename="telegram_voice.mp3",
                    content_type="audio/mp3",
                    file_content=mp3_output_io.getvalue()
                )

                await message.reply_text("Transcribing audio...", quote=True)
                transcribed_text_result = await _transcribe_audio_function(mock_upload_file)
                await message.reply_text(f"Transcription: {transcribed_text_result}")

                await message.reply_text("Fact-checking transcription...", quote=True)
                fact_check_result = await _perform_text_factcheck(transcribed_text_result)
                await message.reply_text(fact_check_result)
                
                if _tts_function:
                    try:
                        audio_bytes_base64 = _tts_function(fact_check_result) # This now returns base64 string
                        audio_bytes = base64.b64decode(audio_bytes_base64) # Decode back to bytes
                        audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                        ogg_output = io.BytesIO()
                        audio.export(ogg_output, format="ogg", codec="libopus")
                        ogg_output.seek(0)
                        await context.bot.send_voice(chat_id=chat_id, voice=ogg_output, caption="Here's the audio fact-check:")
                    except Exception as audio_e:
                        print(f"Error sending audio for voice fact-check: {audio_e}")
                        await message.reply_text("Could not send audio response for the fact-check.")

            except Exception as e:
                print(f"Error processing voice message: {e}")
                await message.reply_text(f"An error occurred while processing your voice message: {e}. Please try again.")
            finally:
                voice_ogg_bytes_io.close()
                mp3_output_io.close()


        elif message.photo and _perform_image_factcheck_function:
            file_id = message.photo[-1].file_id # Get the largest photo
            image_file = await context.bot.get_file(file_id)
            
            image_bytes_io = io.BytesIO()
            await image_file.download_to_memory(image_bytes_io)
            image_bytes_io.seek(0) # Rewind to the beginning

            caption = message.caption if message.caption else ""
            await message.reply_text("Fact-checking your image...", quote=True)
            
            # The perform_image_factcheck_function now expects bytes, mime_type, and caption
            # We need to guess the mime type here for the telegram bot.
            # Use imghdr for robust detection
            img_bytes = image_bytes_io.getvalue()
            img_type = imghdr.what(None, h=img_bytes)
            mime_type = f"image/{img_type}" if img_type else "application/octet-stream" # Fallback

            fact_check_response = await _perform_image_factcheck_function(img_bytes, mime_type, caption)
            
            if "error" in fact_check_response:
                await message.reply_text(f"Error during image fact-check: {fact_check_response['error']}")
            else:
                await message.reply_text(fact_check_response['result'])

                if _tts_function:
                    try:
                        audio_bytes_base64 = _tts_function(fact_check_response['result']) # This now returns base64 string
                        audio_bytes = base64.b64decode(audio_bytes_base64) # Decode back to bytes
                        audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                        ogg_output = io.BytesIO()
                        audio.export(ogg_output, format="ogg", codec="libopus")
                        ogg_output.seek(0)
                        await context.bot.send_voice(chat_id=chat_id, voice=ogg_output)
                    except Exception as audio_e:
                        print(f"Error sending audio for image fact-check: {audio_e}")
                        await message.reply_text("Could not send audio response for the image fact-check.")
        else:
            await message.reply_text("Sorry, I can only fact-check text, voice messages, and photos.")
            
    except Exception as e:
        print(f"Error handling Telegram message: {e}")
        await message.reply_text(f"An error occurred: {e}. Please try again.")

def setup_telegram_bot_application(token: str) -> Application:
    """
    Sets up the Telegram Application and registers all handlers.
    """
    application = Application.builder().token(token).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_message))
    
    return application
