// DOM Elements
const resultOutput = document.getElementById("resultOutput");
const transcriptionBox = document.getElementById("transcriptionBox");
const transcriptionText = document.getElementById("transcriptionText"); // Get the p element
const loadingSpinner = document.getElementById("loadingSpinner");
const recordBtn = document.getElementById("recordBtn");
const recordLabel = document.getElementById("recordLabel");
const audioPlayer = document.getElementById("player"); // Renamed for clarity

const BASE_URL = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "https://proverify.onrender.com"; // <-- Update this if your Render URL changes!

const modes = ["voiceMode", "textMode", "imageMode"];

function showMode(modeId) {
    modes.forEach(mode => {
        document.getElementById(mode).style.display = (mode === modeId) ? "block" : "none";
    });
    resultOutput.textContent = "Fact checked results will appear here..."; // Reset result text
    transcriptionText.textContent = "(No transcription yet)"; // Reset transcription text
    transcriptionBox.style.display = "none"; // Hide transcription box initially
    // Also stop any currently playing audio when switching modes
    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.src = "";
        audioPlayer.style.display = "none";
    }
}

// TEXT FACT CHECKING
async function submitText() {
    const input = document.getElementById("inputText").value.trim();
    if (!input) {
        // Using custom modal/message box instead of alert()
        showMessageBox("Please enter some text to fact-check.");
        return;
    }

    resultOutput.textContent = "Checking...";
    loadingSpinner.style.display = "inline-block";

    try {
        const response = await fetch(`${BASE_URL}/factcheck/text`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: input }),
        });

        const data = await response.json();
        resultOutput.textContent = data.result || data.error || "Unknown error";

        // Play the audio result if available
        if (data.audio_result) {
            playBase64Audio(data.audio_result);
        } else if (data.error) {
            showMessageBox(`Error: ${data.error}`);
        }

    } catch (err) {
        console.error("Text fact-check error:", err);
        resultOutput.textContent = "Unable to process text. Network error or server issue.";
        showMessageBox("Failed to process text. Please try again.");
    } finally {
        loadingSpinner.style.display = "none";
    }
}

// AUDIO FILE UPLOAD
async function uploadAudio() {
    const fileInput = document.getElementById("audioInput");
    const file = fileInput.files[0];
    if (!file) {
        showMessageBox("Please select an audio file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    resultOutput.textContent = "Uploading & checking...";
    loadingSpinner.style.display = "inline-block";
    transcriptionBox.style.display = "none"; // Hide transcription box while uploading

    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => {
            controller.abort();
            resultOutput.textContent = "Request timed out. Please try again.";
            loadingSpinner.style.display = "none";
            showMessageBox("Audio upload timed out. Please try again.");
        }, 30000); // 30 seconds timeout

        const response = await fetch(`${BASE_URL}/factcheck/audio`, {
            method: "POST",
            body: formData,
            signal: controller.signal,
        });
        clearTimeout(timeout);

        const data = await response.json();
        if (data.transcription) {
            transcriptionText.textContent = data.transcription;
            transcriptionBox.style.display = "block"; // Show transcription box
        } else {
            transcriptionText.textContent = "(No transcription)";
            transcriptionBox.style.display = "block";
        }
        resultOutput.textContent = data.result || data.error || "Unknown error";

        // Play the audio result if available
        if (data.audio_result) {
            playBase64Audio(data.audio_result);
        } else if (data.error) {
            showMessageBox(`Error: ${data.error}`);
        }

    } catch (err) {
        console.error("Audio upload error:", err);
        if (err.name === 'AbortError') {
            resultOutput.textContent = "Audio upload timed out. Please try again.";
        } else {
            resultOutput.textContent = "Unable to process audio. Network error or server issue.";
        }
        showMessageBox("Failed to process audio. Please try again.");
    } finally {
        loadingSpinner.style.display = "none";
    }
}

// AUDIO RECORDING SETUP
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

async function toggleRecording() {
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

            mediaRecorder.onstop = async () => {
                const blob = new Blob(audioChunks, { type: "audio/webm" });
                audioPlayer.src = URL.createObjectURL(blob);
                audioPlayer.style.display = "block"; // Show the audio player

                const formData = new FormData();
                formData.append("file", blob, "recording.webm");

                resultOutput.textContent = "Processing recording...";
                loadingSpinner.style.display = "inline-block";
                transcriptionBox.style.display = "none"; // Hide transcription box initially

                try {
                    const response = await fetch(`${BASE_URL}/factcheck/audio`, {
                        method: "POST",
                        body: formData,
                    });
                    const data = await response.json();
                    if (data.transcription) {
                        transcriptionText.textContent = data.transcription;
                        transcriptionBox.style.display = "block"; // Show transcription box
                    } else {
                        transcriptionText.textContent = "(No transcription)";
                        transcriptionBox.style.display = "block";
                    }
                    resultOutput.textContent = data.result || data.error || "Unknown error";

                    // Play the audio result if available
                    if (data.audio_result) {
                        playBase64Audio(data.audio_result);
                    } else if (data.error) {
                        showMessageBox(`Error: ${data.error}`);
                    }

                } catch (err) {
                    console.error("Recorded audio processing error:", err);
                    resultOutput.textContent = "Unable to process recorded audio. Network error or server issue.";
                    showMessageBox("Failed to process recorded audio. Please try again.");
                } finally {
                    loadingSpinner.style.display = "none";
                }
            };

            mediaRecorder.start();
            isRecording = true;
            recordBtn.textContent = "⏹";
            recordLabel.textContent = "Stop recording";
        } catch (err) {
            console.error("Error accessing microphone:", err);
            showMessageBox("Could not access microphone. Please ensure it's connected and permissions are granted.");
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.textContent = "🎤";
        recordLabel.textContent = "Click to start recording";
    }
}

function removeAudioFile() {
    document.getElementById("audioInput").value = "";
    audioPlayer.style.display = "none";
    audioPlayer.src = ""; // Clear the audio source
    transcriptionText.textContent = "(No transcription yet)";
    transcriptionBox.style.display = "none";
    resultOutput.textContent = "Fact checked results will appear here...";
}

// IMAGE FACT CHECKING
async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const caption = document.getElementById("captionInput").value;
    const file = fileInput.files[0];

    if (!file) {
        showMessageBox("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("caption", caption);

    resultOutput.textContent = "Checking image...";
    loadingSpinner.style.display = "inline-block";

    try {
        const response = await fetch(`${BASE_URL}/factcheck/image`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        resultOutput.textContent = data.result || data.error || "Unknown error";

        // --- ADDED: Play the audio result for image fact-check ---
        if (data.audio_result) {
            playBase64Audio(data.audio_result);
        } else if (data.error) {
            showMessageBox(`Error: ${data.error}`);
        }
        // --- END ADDED ---

    } catch (err) {
        console.error("Image upload error:", err);
        resultOutput.textContent = "Unable to process image. Network error or server issue.";
        showMessageBox("Failed to process image. Please try again.");
    } finally {
        loadingSpinner.style.display = "none";
    }
}

function removeImageFile() {
    document.getElementById("imageInput").value = "";
    // If you had an image preview, you'd clear it here
    resultOutput.textContent = "Fact checked results will appear here...";
    // Stop any playing audio when removing the image
    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.src = "";
        audioPlayer.style.display = "none";
    }
}

// Helper function to play base64 audio
function playBase64Audio(base64Audio) {
    try {
        // Ensure the audio player is visible if it was hidden
        audioPlayer.style.display = "block";
        audioPlayer.src = "data:audio/mp3;base64," + base64Audio;
        audioPlayer.play().catch(e => console.error("Audio playback error:", e));
    } catch (e) {
        console.error("Failed to create audio object:", e);
    }
}

// Custom message box function (replaces alert())
function showMessageBox(message) {
    // A simple, non-blocking way to show messages.
    // For a more robust solution, you'd create a proper modal dialog.
    const messageBox = document.createElement('div');
    messageBox.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 15px 25px;
        border-radius: 8px;
        z-index: 1000;
        font-family: 'Segoe UI', sans-serif;
        font-size: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        gap: 10px;
    `;
    messageBox.innerHTML = `<span>${message}</span>`;
    document.body.appendChild(messageBox);

    setTimeout(() => {
        messageBox.remove();
    }, 3000); // Message disappears after 3 seconds
}


// Initialize the first mode on page load
document.addEventListener("DOMContentLoaded", () => {
    showMode('voiceMode');
});
