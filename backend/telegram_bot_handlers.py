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
_perform_image_factcheck_function = None

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
    if not _openai_client:
        return "Fact-checking service is not available. OpenAI client not initialized."
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
    ai_response = _openai_client.chat.completions.create(
        model="o4-mini", # Reverted to original model name
        messages=messages
    )
    fact_check_result = ai_response.choices[0].message.content
    audio_base64 = _tts_function(fact_check_result) if _tts_function else None
    return {"result": fact_check_result, "audio_result": audio_base64}

async def _perform_audio_factcheck_bot(audio_file_path: str):
    """Performs audio transcription and fact-check using the shared OpenAI client."""
    if not _openai_client:
        return {"transcription": "Error: OpenAI client not initialized.", "result": "Error", "audio_result": None}

    with open(audio_file_path, "rb") as audio_file:
        transcript = _openai_client.audio.transcriptions.create(
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

    ai_response = _openai_client.chat.completions.create(
        model="o4-mini", # Reverted to original model name
        messages=messages
    )
    fact_check_result = ai_response.choices[0].message.content
    audio_base64_result = _tts_function(fact_check_result) if _tts_function else None
    return {"transcription": transcribed_text, "result": fact_check_result, "audio_result": audio_base64_result}


# --- TELEGRAM BOT COMMAND HANDLERS ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message when the command /start is issued."""
    user_id = update.effective_user.id
    if is_authorized(user_id):
        await update.message.reply_text(
            f"Hello {update.effective_user.first_name}! 👋\n"
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
            "Here's what I can do:\n"
            "- Send me *any text message* to fact-check it.\n"
            "- Send me a *voice message* and I'll transcribe and fact-check it.\n"
            "- Send me a *photo* and I'll analyze it for misleading information.\n"
            "I'm here to help you verify information!"
            , parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text("You are not authorized to use this bot.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text, voice, and photo messages."""
    message = update.message
    chat_id = message.chat_id
    user_id = message.from_user.id

    if not is_authorized(user_id):
        await message.reply_text("You are not authorized to use this bot.")
        return

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
            # Using tempfile to handle file properly
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                tmp.write(voice_bytes)
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

        elif message.photo and _perform_image_factcheck_function:
            await message.reply_text("Processing photo...")
            photo_file_id = message.photo[-1].file_id # Get the largest photo size
            photo_file = await context.bot.get_file(photo_file_id)
            photo_bytes = await photo_file.download_as_bytearray()
            
            caption = message.caption if message.caption else ""

            # Call the perform_image_factcheck_func (from app.py) which is now the unified Gemini function
            # The function expects base64 encoded image data directly.
            image_base64 = base64.b64encode(photo_bytes).decode('utf-8')
            
            response = await _perform_image_factcheck_function(image_data=bytes(photo_bytes), caption=caption)
            
            # Extract result, audio, and sources from the Gemini response
            fact_check_result = response # Assuming this directly returns the string result
            audio_base64 = None # Assuming image fact-check does not return audio from app.py directly
            # If app.py's perform_image_factcheck also returns audio, you'd extract it here.

            response_text = f"✅ Image Fact Check: {fact_check_result}"
            # If sources were part of the response, they'd be added here
            # if sources:
            #     response_text += "\n\nSources:\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(sources)])

            await message.reply_text(response_text)

            # Send the audio result if available (assuming _tts_function can generate audio from the text result)
            if _tts_function and fact_check_result:
                try:
                    audio_bytes = _tts_function(fact_check_result)
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
