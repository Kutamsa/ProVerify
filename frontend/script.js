// DOM Elements
const resultOutput = document.getElementById("resultOutput");
const transcriptionBox = document.getElementById("transcriptionBox");
const transcriptionText = document.getElementById("transcriptionText");
const loadingSpinner = document.getElementById("loadingSpinner");
const recordBtn = document.getElementById("recordBtn");
const recordLabel = document.getElementById("recordLabel");
const audioPlayer = document.getElementById("player");

// NEW DOM Elements for News Feed
const rssUrlInput = document.getElementById("rssUrlInput");
const sourceNameInput = document.getElementById("sourceNameInput");
const addSourceBtn = document.getElementById("addSourceBtn");
const articlesList = document.getElementById("articlesList");
const newsArticlesContainer = document.getElementById("newsArticlesContainer");
const newsSourceButtonsContainer = document.getElementById("newsSourceButtonsContainer"); // NEW
const loadMoreBtn = document.getElementById("loadMoreBtn"); // NEW

// Message box element - now assumed to be in index.html
const messageBox = document.getElementById('messageBox');


const BASE_URL = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "https://proverify.onrender.com"; // <-- Update this if your Render URL changes!

const modes = ["voiceMode", "textMode", "imageMode"];

let mediaRecorder;
let audioChunks = [];
let activeSourceId = null; // NEW: To keep track of the currently selected source
let currentPage = 0; // For pagination of news articles
const ARTICLES_PER_PAGE = 10; // Consistent with backend limit

// Helper function to display messages to the user
function showMessageBox(message, isError = false) {
    messageBox.textContent = message;
    // Updated colors to match the palette derived from the image (red and green for messages)
    messageBox.style.backgroundColor = isError ? "#EF4444" : "#22C55E"; // Red for error, Green for success
    messageBox.style.display = "block";
    setTimeout(() => {
        messageBox.style.opacity = 1;
    }, 10); // Small delay for CSS transition

    setTimeout(() => {
        messageBox.style.opacity = 0;
        messageBox.addEventListener('transitionend', function handler() {
            messageBox.style.display = 'none';
            messageBox.removeEventListener('transitionend', handler);
        });
    }, 5000); // Hide after 5 seconds
}


function showMode(modeId) {
    modes.forEach(mode => {
        const element = document.getElementById(mode);
        if (element) {
            element.style.display = (mode === modeId) ? "block" : "none";
        }
    });

    // Stop recording and audio playback if active
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        recordBtn.textContent = "🎤"; // Ensure mic symbol
        recordBtn.classList.remove("recording");
        recordLabel.textContent = "Click to Record";
    }
    if (!audioPlayer.paused) {
        audioPlayer.pause();
        audioPlayer.src = ''; // Clear audio source
    }
    loadingSpinner.style.display = "none"; // Hide spinner
    resultOutput.textContent = "Fact checked results will appear here..."; // Reset results
    transcriptionBox.style.display = "none";
    transcriptionText.textContent = "(No transcription yet)";

    // Update active class for buttons
    document.querySelectorAll('.mode-buttons-row button').forEach(button => {
        if (button.onclick.toString().includes(`showMode('${modeId}')`)) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

// --- Voice Mode Functions ---
async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordLabel.textContent = "Processing...";
        recordBtn.classList.remove("recording");
        recordBtn.textContent = "⏹️"; // Change to stop symbol
    } else {
        audioChunks = [];
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/mp3" });
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayer.src = audioUrl;

                loadingSpinner.style.display = "block";
                resultOutput.textContent = ""; // Clear previous results
                transcriptionBox.style.display = "none"; // Hide transcription during upload
                recordLabel.textContent = "Processing..."; // Keep processing message

                const formData = new FormData();
                formData.append("audio_file", audioBlob, "recording.mp3");

                try {
                    const response = await fetch(`${BASE_URL}/factcheck/audio`, {
                        method: "POST",
                        body: formData,
                    });
                    const data = await response.json();
                    loadingSpinner.style.display = "none"; // Hide spinner after process completion
                    recordLabel.textContent = "Click to Record"; // Clear processing message

                    if (response.ok) {
                        transcriptionText.textContent = data.transcription; // Correct key
                        transcriptionBox.style.display = "block";
                        resultOutput.textContent = data.result; // FIX: Changed from data.factCheckResult to data.result
                        if (data.audio_result) { // FIX: Changed from data.audio to data.audio_result
                            const audioBytes = Uint8Array.from(atob(data.audio_result), c => c.charCodeAt(0));
                            const blob = new Blob([audioBytes], { type: "audio/mp3" });
                            const url = URL.createObjectURL(blob);
                            audioPlayer.src = url;
                            audioPlayer.play();
                        }
                    } else {
                        resultOutput.textContent = `Error: ${data.error || "Unknown error"}`;
                        showMessageBox(`Error: ${data.error || "Unknown error"}`, true);
                    }
                } catch (error) {
                    console.error("Error during audio fact-check:", error);
                    loadingSpinner.style.display = "none"; // Hide spinner
                    recordLabel.textContent = "Click to Record"; // Clear processing message
                    resultOutput.textContent = "Error: Could not connect to server.";
                    showMessageBox("Failed to connect to server.", true);
                }
            };
            mediaRecorder.start();
            recordLabel.textContent = "Recording...";
            recordBtn.classList.add("recording");
            recordBtn.textContent = "⏹️"; // Change to stop symbol
        } catch (error) {
            console.error("Error accessing microphone:", error);
            recordLabel.textContent = "Error: Microphone access denied.";
            showMessageBox("Microphone access denied. Please allow microphone access in your browser settings.", true);
        }
    }
}

// --- Text Mode Functions ---
async function submitText() {
    const inputText = document.getElementById("inputText").value;
    if (!inputText.trim()) {
        showMessageBox("Please enter some text to fact-check.", true);
        return;
    }

    loadingSpinner.style.display = "block";
    resultOutput.textContent = ""; // Clear previous results

    try {
        const response = await fetch(`${BASE_URL}/factcheck/text`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: inputText }),
        });
        const data = await response.json();
        loadingSpinner.style.display = "none"; // Hide spinner after process completion

        if (response.ok) {
            resultOutput.textContent = data.result; // FIX: Changed from data.factCheckResult to data.result
            if (data.audio_result) { // FIX: Changed from data.audio to data.audio_result
                const audioBytes = Uint8Array.from(atob(data.audio_result), c => c.charCodeAt(0));
                const blob = new Blob([audioBytes], { type: "audio/mp3" });
                const url = URL.createObjectURL(blob);
                audioPlayer.src = url;
                audioPlayer.play();
            }
        } else {
            resultOutput.textContent = `Error: ${data.error || "Unknown error"}`;
            showMessageBox(`Error: ${data.error || "Unknown error"}`, true);
        }
    } catch (error) {
        console.error("Error during text fact-check:", error);
        loadingSpinner.style.display = "none"; // Hide spinner
        resultOutput.textContent = "Error: Could not connect to server.";
        showMessageBox("Failed to connect to server.", true);
    }
}

// --- Image Mode Functions ---
function removeImageFile() {
    document.getElementById('imageInput').value = '';
    document.getElementById('captionInput').value = '';
    showMessageBox("Image and caption cleared.", false);
}

async function uploadImage() {
    const imageInput = document.getElementById("imageInput");
    const captionInput = document.getElementById("captionInput");
    const imageFile = imageInput.files[0];
    const caption = captionInput.value.trim();

    if (!imageFile) {
        showMessageBox("Please select an image file to upload.", true);
        return;
    }

    loadingSpinner.style.display = "block";
    resultOutput.textContent = ""; // Clear previous results

    const formData = new FormData();
    formData.append("image", imageFile);
    if (caption) {
        formData.append("caption", caption);
    }

    try {
        const response = await fetch(`${BASE_URL}/factcheck/image`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        loadingSpinner.style.display = "none"; // Hide spinner after process completion

        if (response.ok) {
            resultOutput.textContent = data.result; // FIX: Changed from data.factCheckResult to data.result
            if (data.audio_result) { // FIX: Changed from data.audio to data.audio_result
                const audioBytes = Uint8Array.from(atob(data.audio_result), c => c.charCodeAt(0));
                const blob = new Blob([audioBytes], { type: "audio/mp3" });
                const url = URL.createObjectURL(blob);
                audioPlayer.src = url;
                audioPlayer.play();
            }
        } else {
            resultOutput.textContent = `Error: ${data.error || "Unknown error"}`;
            showMessageBox(`Error: ${data.error || "Unknown error"}`, true);
        }
    } catch (error) {
        console.error("Error during image fact-check:", error);
        loadingSpinner.style.display = "none"; // Hide spinner
        resultOutput.textContent = "Error: Could not connect to server.";
        showMessageBox("Failed to connect to server.", true);
    }
}

// --- News Feed Functions ---
async function addNewsSource() {
    const url = rssUrlInput.value.trim();
    const name = sourceNameInput.value.trim();

    if (!url || !name) {
        showMessageBox("Please enter both URL/RSS feed and Source Name.", true);
        return;
    }

    try {
        const formData = new FormData();
        formData.append("source_url", url);
        formData.append("source_name", name);

        const response = await fetch(`${BASE_URL}/news/add_source`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (response.ok) {
            showMessageBox("Source added successfully!");
            rssUrlInput.value = "";
            sourceNameInput.value = "";
            await loadNewsSources(); // Reload sources to display the new one

            // After adding a new source, select it to trigger article loading
            // The backend returns the ID of the newly added source, so we can use that.
            if (data.id) {
                await selectNewsSource(data.id);
            } else {
                // Fallback: If for some reason ID is not returned, just re-fetch all news.
                fetchNewsArticles();
            }
        } else {
            showMessageBox(`Error: ${data.error || "Failed to add source."}`, true);
        }
    } catch (error) {
        console.error("Error adding news source:", error);
        showMessageBox("Failed to connect to server to add source.", true);
    }
}

// Function to attach event listeners to remove buttons
function attachRemoveSourceListener(button, sourceId) {
    button.onclick = async () => {
        // IMPORTANT: Avoid using window.confirm directly in Canvas environments
        // Consider a custom modal dialog for better user experience.
        if (!confirm("Are you sure you want to remove this source and all its articles?")) {
            return;
        }
        try {
            const formData = new FormData();
            formData.append("source_id", sourceId);

            const response = await fetch(`${BASE_URL}/news/remove_source`, {
                method: "POST",
                body: formData,
            });
            const data = await response.json();

            if (response.ok) {
                showMessageBox("Source removed successfully!");
                loadNewsSources(); // Reload remaining sources
                fetchNewsArticles(); // Refresh general news list
            } else {
                showMessageBox(`Error: ${data.error || "Failed to remove source."}`, true);
            }
        } catch (error) {
            console.error("Error removing news source:", error);
            showMessageBox("Failed to connect to server to remove source.", true);
        }
    };
}


async function loadNewsSources() {
    try {
        const response = await fetch(`${BASE_URL}/news/sources`);
        const data = await response.json();

        newsSourceButtonsContainer.innerHTML = ''; // Clear existing buttons

        if (response.ok && data.sources && data.sources.length > 0) {
            data.sources.forEach(source => {
                const sourceButton = document.createElement('div');
                sourceButton.className = 'source-button';
                sourceButton.setAttribute('data-source-id', source.id); // Store ID

                const buttonText = document.createElement('span');
                buttonText.textContent = source.name;
                buttonText.onclick = () => {
                    selectNewsSource(source.id);
                };

                const removeBtn = document.createElement('button');
                removeBtn.className = 'remove-source-btn';
                removeBtn.innerHTML = '&#x2716;'; // Cross icon
                attachRemoveSourceListener(removeBtn, source.id); // Attach listener with ID

                sourceButton.appendChild(buttonText);
                sourceButton.appendChild(removeBtn);
                newsSourceButtonsContainer.appendChild(sourceButton);
            });
        } else {
            newsSourceButtonsContainer.innerHTML = '<p>No news sources added yet.</p>';
        }
    } catch (error) {
        console.error("Error loading news sources:", error);
        showMessageBox("Failed to load news sources.", true);
    }
}

async function selectNewsSource(sourceId) {
    activeSourceId = sourceId;
    currentPage = 0; // Reset page for new source
    articlesList.innerHTML = ''; // Clear current articles
    loadMoreBtn.style.display = 'none'; // Hide load more until articles are loaded

    // Visually mark the active source button
    document.querySelectorAll('.source-button').forEach(btn => {
        if (parseInt(btn.getAttribute('data-source-id')) === sourceId) {
            btn.classList.add('active-source');
        } else {
            btn.classList.remove('active-source');
        }
    });

    // Optionally fetch and store latest for the selected source before displaying
    try {
        const response = await fetch(`${BASE_URL}/news/fetch_and_store?source_id=${sourceId}`, {
            method: 'POST'
        });
        const data = await response.json();
        if (response.ok) {
            console.log(data.message); // Log success
            showMessageBox(data.message);
        } else {
            console.error("Error fetching and storing new articles for source:", data.error);
            showMessageBox(`Error updating news for source: ${data.error}`, true);
        }
    }
    catch (error) {
        console.error("Network error during fetch_and_store:", error);
        showMessageBox("Network error updating news for source.", true);
    }
    // Always fetch articles after trying to update, even if update failed
    fetchNewsArticles(sourceId, false); // Fetch and display for the selected source, not appending
}

async function fetchNewsArticles(sourceId = null, append = false) {
    loadingSpinner.style.display = "block"; // Show loading for articles fetch
    if (!append) {
        articlesList.innerHTML = '<li>Loading articles...</li>'; // Clear only if not appending
    }

    let url = `${BASE_URL}/news/articles?offset=${currentPage * ARTICLES_PER_PAGE}&limit=${ARTICLES_PER_PAGE}`;
    if (sourceId) {
        url += `&source_id=${sourceId}`;
    }

    try {
        const response = await fetch(url);
        const data = await response.json();
        loadingSpinner.style.display = "none"; // Hide spinner after process completion

        if (response.ok) {
            if (!append) {
                articlesList.innerHTML = ''; // Clear again if not appending
            }

            if (data.articles && data.articles.length > 0) {
                data.articles.forEach(article => {
                    const listItem = document.createElement('li');
                    listItem.innerHTML = `
                        <a href="${article.link}" target="_blank" rel="noopener noreferrer">${article.title}</a>
                        ${article.source ? `<span> - ${article.source}</span>` : ''}
                        ${article.pubDate ? `<span> (${new Date(article.pubDate).toLocaleDateString()})</span>` : ''}
                    `;
                    articlesList.appendChild(listItem);
                });
                currentPage++; // Increment page for next load more
                if (data.hasMore) {
                    loadMoreBtn.style.display = 'block';
                } else {
                    loadMoreBtn.style.display = 'none';
                    if (articlesList.innerHTML === '') { // If no articles after pagination, show message
                        articlesList.innerHTML = '<li>No more articles found for this selection.</li>';
                    }
                }
            } else if (!append) { // No articles found and not appending
                articlesList.innerHTML = '<li>No news articles found for this selection.</li>';
                loadMoreBtn.style.display = 'none';
            } else { // No more articles to append
                loadMoreBtn.style.display = 'none';
                showMessageBox("No more articles to load.");
            }
        } else {
            articlesList.innerHTML = `<li>Error fetching news: ${data.error || "Unknown error"}</li>`;
            showMessageBox(`Error fetching news: ${data.error || "Unknown error"}`, true);
            loadMoreBtn.style.display = 'none';
        }
    } catch (err) {
        loadingSpinner.style.display = "none"; // Hide spinner
        console.error("Fetch news articles error:", err);
        articlesList.innerHTML = '<li>Unable to load news articles. Network error or server issue.</li>';
        showMessageBox("Failed to fetch news articles. Please try again.", true);
    }
}

// Event listener for Load More button
loadMoreBtn.addEventListener('click', () => {
    fetchNewsArticles(activeSourceId, true); // Append articles for the current source
});

// Initialize the first mode and fetch news on page load
document.addEventListener("DOMContentLoaded", () => {
    showMode('voiceMode'); // Or your preferred default mode
    loadNewsSources(); // Load news source buttons on page load
    fetchNewsArticles(); // Fetch all news when the page loads initially
});
