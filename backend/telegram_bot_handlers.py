import os
import io
import base64
import tempfile
from pydub import AudioSegment
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# These global variables will be set during the bot's initialization
# by the main app.py file.
_openai_client = None
_tts_function = None
_authorized_user_ids = []
_perform_image_factcheck_function = None # To avoid circular imports if app.py imports us

def initialize_bot_components(
    openai_client_instance,
    tts_func,
    authorized_ids,
    perform_image_factcheck_func
):
    """
    Initializes the necessary components for the Telegram bot handlers.
    This function should be called once from app.py on startup.
    """
    global _openai_client, _tts_function, _authorized_user_ids, _perform_image_factcheck_function
    _openai_client = openai_client_instance
    _tts_function = tts_func
    _authorized_user_ids = authorized_ids
    _perform_image_factcheck_function = perform_image_factcheck_func
    print("Telegram bot components initialized.")

def is_authorized(user_id: int) -> bool:
    """Checks if a user is authorized to use the bot."""
    return user_id in _authorized_user_ids

async def _perform_text_factcheck_bot(input_text: str):
    """Performs text fact-check using the shared OpenAI client."""
    prompt = f"Fact-check this text and provide a clear answer:\n\n{input_text}"
    ai_response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a helpful fact-checking assistant. Respond clearly and concisely."},
            {"role": "user", "content": prompt}
        ]
    )
    fact_check_result = ai_response.choices[0].message.content
    audio_base64 = _tts_function(fact_check_result)
    return {"result": fact_check_result, "audio_result": audio_base64}

async def _perform_audio_factcheck_bot(audio_file_path: str):
    """Performs audio transcription and fact-check using the shared OpenAI client."""
    with open(audio_file_path, "rb") as audio_file:
        transcript = _openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        transcribed_text = transcript.text

    prompt = f"Fact-check this audio content (transcribed in Telugu):\n\n{transcribed_text}"
    ai_response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a Telugu fact-checking assistant. Provide clear and concise answers in Telugu."},
            {"role": "user", "content": prompt}
        ]
    )
    fact_check_result = ai_response.choices[0].message.content
    audio_base64_result = _tts_function(fact_check_result)
    return {"transcription": transcribed_text, "result": fact_check_result, "audio_result": audio_base64_result}


# --- TELEGRAM BOT COMMAND HANDLERS ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message when the command /start is issued."""
    user_id = update.effective_user.id
    if is_authorized(user_id):
        await update.message.reply_html(
            f"Hello {update.effective_user.mention_html()}! 👋\n"
            "I am your Fact-Checking bot. Send me text, a voice message, or a photo to fact-check it. "
            "You can also use /factcheck_text <your_text>."
        )
    else:
        await update.message.reply_text("You are not authorized to use this bot. Please contact the administrator.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message when the command /help is issued."""
    user_id = update.effective_user.id
    if is_authorized(user_id):
        await update.message.reply_text(
            "Send me text to fact-check.\n"
            "Send me a voice message to get a transcription and fact-check.\n"
            "Send me a photo with an optional caption to fact-check the image.\n"
            "Use /factcheck_text <your_text> for direct text fact-checking."
        )
    else:
        await update.message.reply_text("You are not authorized to use this bot.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles various types of messages (text, voice, photo)."""
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    message = update.message
    chat_id = message.chat_id

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        if message.text:
            if message.text.startswith('/factcheck_text '):
                input_text = message.text[len('/factcheck_text '):].strip()
                if not input_text:
                    await message.reply_text("Please provide text after /factcheck_text.")
                    return
                await message.reply_text("Checking text...")
                response = await _perform_text_factcheck_bot(input_text)
                await message.reply_text(f"✅ Fact Check: {response.get('result', 'Error fetching result.')}")
                if response.get('audio_result'):
                    audio_bytes = base64.b64decode(response['audio_result'])
                    audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                    ogg_output = io.BytesIO()
                    audio.export(ogg_output, format="ogg", codec="libopus")
                    ogg_output.seek(0)
                    await context.bot.send_voice(chat_id=chat_id, voice=ogg_output)
            else: # Default text message handling
                await message.reply_text("Checking text...")
                response = await _perform_text_factcheck_bot(message.text)
                await message.reply_text(f"✅ Fact Check: {response.get('result', 'Error fetching result.')}")
                if response.get('audio_result'):
                    audio_bytes = base64.b64decode(response['audio_result'])
                    audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                    ogg_output = io.BytesIO()
                    audio.export(ogg_output, format="ogg", codec="libopus")
                    ogg_output.seek(0)
                    await context.bot.send_voice(chat_id=chat_id, voice=ogg_output)

        elif message.voice:
            await message.reply_text("Processing voice message...")
            voice_file = await message.voice.get_file()
            voice_bytes = await voice_file.download_as_bytearray()

            # Telegram voice messages are typically OGG Opus. Convert to webm for Whisper.
            input_audio = AudioSegment.from_file(io.BytesIO(voice_bytes), format="ogg")
            webm_output = io.BytesIO()
            input_audio.export(webm_output, format="webm")
            webm_output.seek(0) # Rewind to start of stream

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(webm_output.getvalue())
                tmp_path = tmp.name

            try:
                response = await _perform_audio_factcheck_bot(tmp_path)
                transcription = response.get('transcription', 'No transcription found.')
                fact_check = response.get('result', 'Error fetching result.')
                await message.reply_text(f"🗣️ Transcription: {transcription}\n\n✅ Fact Check: {fact_check}")
                if response.get('audio_result'):
                    audio_bytes = base64.b64decode(response['audio_result'])
                    audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                    ogg_output = io.BytesIO()
                    audio.export(ogg_output, format="ogg", codec="libopus")
                    ogg_output.seek(0)
                    await context.bot.send_voice(chat_id=chat_id, voice=ogg_output)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        elif message.photo:
            await message.reply_text("Processing photo...")
            photo_file_id = message.photo[-1].file_id # Get the largest photo size
            photo_file = await context.bot.get_file(photo_file_id)
            photo_bytes = await photo_file.download_as_bytearray()

            caption = message.caption if message.caption else ""
            mime_type = "image/jpeg" # Common type for Telegram photos

            response = await _perform_image_factcheck_function(bytes(photo_bytes), mime_type, caption)
            await message.reply_text(f"✅ Image Fact Check: {response.get('result', 'Error fetching result.')}")

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

