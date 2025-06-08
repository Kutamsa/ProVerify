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

const BASE_URL = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "https://proverify.onrender.com"; // <-- Update this if your Render URL changes!

const modes = ["voiceMode", "textMode", "imageMode"]; // newsFeedMode is not a separate display mode, it's a section

let mediaRecorder;
let audioChunks = [];
let activeSourceId = null; // NEW: To keep track of the currently selected source
let currentPage = 0; // NEW: For pagination
const articlesPerPage = 10; // NEW: Limit articles to 10 per page

function showMode(modeId) {
    modes.forEach(mode => {
        const element = document.getElementById(mode);
        if (element) {
            element.style.display = (mode === modeId) ? "block" : "none";
        }
    });
}

function showMessageBox(message) {
    const messageBox = document.createElement("div");
    messageBox.className = "message-box";
    messageBox.textContent = message;
    document.body.appendChild(messageBox);

    setTimeout(() => {
        messageBox.remove();
    }, 3000);
}

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                uploadAudio(audioBlob);
                audioChunks = [];
            };
            mediaRecorder.start();
            recordLabel.textContent = "Recording... Click to stop";
            recordBtn.classList.add("recording");
        })
        .catch(err => {
            console.error("Error accessing microphone:", err);
            showMessageBox("Microphone access denied or error: " + err.message);
        });
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordLabel.textContent = "Processing audio...";
        recordBtn.classList.remove("recording");
    }
}

function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording();
    } else {
        startRecording();
    }
}

async function uploadAudio(audioBlob) {
    loadingSpinner.style.display = "block";
    transcriptionBox.style.display = "none";
    resultOutput.textContent = "Fact checked results will appear here...";
    audioPlayer.style.display = "none";

    const formData = new FormData();
    formData.append("file", audioBlob, "audio.webm");

    try {
        const response = await fetch(`${BASE_URL}/factcheck/audio`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        loadingSpinner.style.display = "none";

        if (response.ok) {
            transcriptionBox.style.display = "block";
            transcriptionText.textContent = data.transcription || "No transcription available.";
            resultOutput.textContent = data.result || "No fact-check result available.";
            if (data.audio_result) {
                audioPlayer.src = `data:audio/mp3;base64,${data.audio_result}`;
                audioPlayer.style.display = "block";
                audioPlayer.play();
            }
        } else {
            resultOutput.textContent = `Error: ${data.error || "Unknown error"}`;
            showMessageBox(`Error: ${data.error || "Unknown error"}`);
        }
    } catch (err) {
        loadingSpinner.style.display = "none";
        console.error("Upload audio error:", err);
        resultOutput.textContent = "Failed to fact-check audio. Network error or server issue.";
        showMessageBox("Failed to fact-check audio. Please try again.");
    }
}

async function submitText() {
    const inputText = document.getElementById("inputText").value;
    if (!inputText.trim()) {
        showMessageBox("Please enter text to fact-check.");
        return;
    }

    loadingSpinner.style.display = "block";
    resultOutput.textContent = "Fact checked results will appear here...";
    audioPlayer.style.display = "none";

    try {
        const response = await fetch(`${BASE_URL}/factcheck/text`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: inputText }),
        });

        const data = await response.json();
        loadingSpinner.style.display = "none";

        if (response.ok) {
            resultOutput.textContent = data.result || "No fact-check result available.";
            if (data.audio_result) {
                audioPlayer.src = `data:audio/mp3;base64,${data.audio_result}`;
                audioPlayer.style.display = "block";
                audioPlayer.play();
            }
        } else {
            resultOutput.textContent = `Error: ${data.error || "Unknown error"}`;
            showMessageBox(`Error: ${data.error || "Unknown error"}`);
        }
    } catch (err) {
        loadingSpinner.style.display = "none";
        console.error("Submit text error:", err);
        resultOutput.textContent = "Failed to fact-check text. Network error or server issue.";
        showMessageBox("Failed to fact-check text. Please try again.");
    }
}

function removeImageFile() {
    const imageInput = document.getElementById("imageInput");
    imageInput.value = ""; // Clear the selected file
}

async function uploadImage() {
    const imageInput = document.getElementById("imageInput");
    const captionInput = document.getElementById("captionInput");
    const imageFile = imageInput.files[0];
    const caption = captionInput.value;

    if (!imageFile) {
        showMessageBox("Please select an image to upload.");
        return;
    }

    loadingSpinner.style.display = "block";
    resultOutput.textContent = "Fact checked results will appear here...";
    audioPlayer.style.display = "none";

    const formData = new FormData();
    formData.append("file", imageFile);
    if (caption.trim()) {
        formData.append("caption", caption);
    }

    try {
        const response = await fetch(`${BASE_URL}/factcheck/image`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        loadingSpinner.style.display = "none";

        if (response.ok) {
            resultOutput.textContent = data.result || "No fact-check result available.";
            if (data.audio_result) {
                audioPlayer.src = `data:audio/mp3;base64,${data.audio_result}`;
                audioPlayer.style.display = "block";
                audioPlayer.play();
            }
        } else {
            resultOutput.textContent = `Error: ${data.error || "Unknown error"}`;
            showMessageBox(`Error: ${data.error || "Unknown error"}`);
        }
    } catch (err) {
        loadingSpinner.style.display = "none";
        console.error("Upload image error:", err);
        resultOutput.textContent = "Failed to fact-check image. Network error or server issue.";
        showMessageBox("Failed to fact-check image. Please try again.");
    }
}

// --- NEWS FEED FUNCTIONS ---

// NEW: Function to load and display news source buttons
async function loadNewsSources() {
    try {
        const response = await fetch(`${BASE_URL}/news/sources`);
        const data = await response.json();

        if (response.ok) {
            newsSourceButtonsContainer.innerHTML = ''; // Clear existing buttons
            if (data.sources && data.sources.length > 0) {
                data.sources.forEach(source => {
                    const sourceButton = document.createElement('button');
                    sourceButton.className = 'source-button';
                    sourceButton.textContent = source.name;
                    sourceButton.dataset.sourceId = source.id; // Store source ID
                    sourceButton.title = source.url; // Show URL on hover

                    // Set active class if this is the currently active source
                    if (activeSourceId === source.id) {
                        sourceButton.classList.add('active');
                    }

                    sourceButton.onclick = () => {
                        // NEW: Set active source and fetch articles for it
                        activeSourceId = source.id;
                        currentPage = 0; // Reset pagination
                        fetchNewsArticles(activeSourceId, false); // Fetch and replace
                        // Remove 'active' from all and add to clicked
                        document.querySelectorAll('.source-button').forEach(btn => btn.classList.remove('active'));
                        sourceButton.classList.add('active');
                    };

                    const removeButton = document.createElement('span');
                    removeButton.className = 'remove-source-btn';
                    removeButton.innerHTML = '&times;'; // 'x' icon
                    removeButton.title = 'Remove source';
                    removeButton.onclick = (event) => {
                        event.stopPropagation(); // Prevent button click event from firing
                        removeNewsSource(source.id);
                    };

                    const buttonWrapper = document.createElement('div');
                    buttonWrapper.className = 'source-button-wrapper';
                    buttonWrapper.appendChild(sourceButton);
                    buttonWrapper.appendChild(removeButton);

                    newsSourceButtonsContainer.appendChild(buttonWrapper);
                });
            } else {
                // Optionally display a message if no sources are added
            }
        } else {
            showMessageBox(`Error fetching sources: ${data.error || "Unknown error"}`);
        }
    } catch (err) {
        console.error("Fetch news sources error:", err);
        showMessageBox("Failed to load news sources. Please try again.");
    }
}

// NEW: Function to add a news source
async function addNewsSource() {
    const rssUrl = rssUrlInput.value.trim();
    const sourceName = sourceNameInput.value.trim();

    if (!rssUrl || !sourceName) {
        showMessageBox("Please enter both URL and Source Name.");
        return;
    }

    try {
        const response = await fetch(`${BASE_URL}/news/add_source`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ url: rssUrl, name: sourceName }),
        });

        const data = await response.json();

        if (response.ok) {
            showMessageBox("Source added successfully!");
            rssUrlInput.value = '';
            sourceNameInput.value = '';
            loadNewsSources(); // Reload sources to display the new button
            // Optionally, select the newly added source and load its articles
            activeSourceId = data.id;
            currentPage = 0;
            fetchNewsArticles(activeSourceId, false);
        } else {
            showMessageBox(`Error adding source: ${data.error || "Unknown error"}`);
        }
    } catch (err) {
        console.error("Add news source error:", err);
        showMessageBox("Failed to add news source. Please try again.");
    }
}

// NEW: Function to remove a news source
async function removeNewsSource(sourceId) {
    if (!confirm('Are you sure you want to remove this news source?')) {
        return;
    }

    try {
        const response = await fetch(`${BASE_URL}/news/remove_source`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ source_id: sourceId }),
        });

        const data = await response.json();

        if (response.ok) {
            showMessageBox("Source removed successfully!");
            loadNewsSources(); // Reload sources
            // If the removed source was active, clear articles and reset activeSourceId
            if (activeSourceId === sourceId) {
                activeSourceId = null;
                currentPage = 0;
                articlesList.innerHTML = '<li>Select a news source or add a new one.</li>';
                loadMoreBtn.style.display = 'none';
            }
        } else {
            showMessageBox(`Error removing source: ${data.error || "Unknown error"}`);
        }
    } catch (err) {
        console.error("Remove news source error:", err);
        showMessageBox("Failed to remove news source. Please try again.");
    }
}


// Modified: Function to fetch news articles with pagination and source filter
async function fetchNewsArticles(sourceId = null, append = false) {
    loadingSpinner.style.display = "block"; // Show loading spinner for articles as well

    if (!append) {
        articlesList.innerHTML = ''; // Clear articles only if not appending
        currentPage = 0; // Reset page if new filter or not appending
    }

    let url = `${BASE_URL}/news/articles?limit=${articlesPerPage}&offset=${currentPage * articlesPerPage}`;
    if (sourceId) {
        url += `&source_id=${sourceId}`;
    }

    try {
        const response = await fetch(url);
        const data = await response.json();
        loadingSpinner.style.display = "none";

        if (response.ok) {
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

                // Show/hide load more button
                if (data.hasMore) {
                    loadMoreBtn.style.display = 'block';
                    currentPage++; // Increment page for next load
                } else {
                    loadMoreBtn.style.display = 'none';
                }

            } else if (!append) { // If no articles found and not appending
                articlesList.innerHTML = '<li>No news articles found for this selection.</li>';
                loadMoreBtn.style.display = 'none';
            } else { // No more articles to append
                loadMoreBtn.style.display = 'none';
                showMessageBox("No more articles to load.");
            }
        } else {
            articlesList.innerHTML = `<li>Error fetching news: ${data.error || "Unknown error"}</li>`;
            showMessageBox(`Error fetching news: ${data.error || "Unknown error"}`);
            loadMoreBtn.style.display = 'none';
        }
    } catch (err) {
        loadingSpinner.style.display = "none";
        console.error("Fetch news articles error:", err);
        articlesList.innerHTML = '<li>Unable to load news articles. Network error or server issue.</li>';
        showMessageBox("Failed to fetch news articles. Please try again.");
        loadMoreBtn.style.display = 'none';
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