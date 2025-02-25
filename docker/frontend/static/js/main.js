// DOM elements
const songInput = document.getElementById('song-input');
const addSongBtn = document.getElementById('add-song-btn');
const songList = document.getElementById('song-list');
const clearSongsBtn = document.getElementById('clear-songs-btn');
const getRecommendationsBtn = document.getElementById('get-recommendations-btn');
const recommendationsCount = document.getElementById('recommendations-count');
const recommendationsList = document.getElementById('recommendations-list');
const ruleRecommendationsList = document.getElementById('rule-recommendations-list');
const similarityRecommendationsList = document.getElementById('similarity-recommendations-list');
const loadingSpinner = document.getElementById('loading');
const errorMessage = document.getElementById('error-message');
const modelInfo = document.getElementById('model-info');
const modelVersion = document.getElementById('model-version');
const modelDate = document.getElementById('model-date');
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

// Array to store selected songs
let selectedSongs = [];

// Function to add a song to the list
function addSong() {
    const songName = songInput.value.trim();
    
    if (songName === '') {
        return;
    }
    
    // Check if song is already in the list
    if (selectedSongs.includes(songName)) {
        showError('This song is already in your list');
        return;
    }
    
    // Add to selectedSongs array
    selectedSongs.push(songName);
    
    // Add to UI
    const li = document.createElement('li');
    li.innerHTML = `
        ${songName}
        <button class="remove-song" data-song="${songName}">Remove</button>
    `;
    songList.appendChild(li);
    
    // Clear input
    songInput.value = '';
    songInput.focus();
    
    // Hide any error message
    hideError();
}

// Function to remove a song from the list
function removeSong(event) {
    if (event.target.classList.contains('remove-song')) {
        const songName = event.target.getAttribute('data-song');
        
        // Remove from array
        selectedSongs = selectedSongs.filter(song => song !== songName);
        
        // Remove from UI
        event.target.parentElement.remove();
    }
}

// Function to clear all songs
function clearSongs() {
    selectedSongs = [];
    songList.innerHTML = '';
}

// Function to switch tabs
function switchTab(event) {
    // Get the tab id
    const tabId = event.target.getAttribute('data-tab');
    
    // Update active button
    tabButtons.forEach(button => {
        if (button.getAttribute('data-tab') === tabId) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // Update active content
    tabContents.forEach(content => {
        if (content.id === `${tabId}-tab`) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

// Function to get recommendations
async function getRecommendations() {
    // Validate
    if (selectedSongs.length === 0) {
        showError('Please add at least one song');
        return;
    }
    
    // Show loading spinner
    loadingSpinner.classList.remove('hidden');
    hideError();
    recommendationsList.innerHTML = '';
    ruleRecommendationsList.innerHTML = '';
    similarityRecommendationsList.innerHTML = '';
    modelInfo.classList.add('hidden');
    
    try {
        // Get selected count
        const count = parseInt(recommendationsCount.value);
        
        // Prepare request data
        const requestData = {
            songs: selectedSongs
        };
        
        // Make API request
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        // Parse response
        const data = await response.json();
        
        // Check for API error
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display recommendations
        displayRecommendations(data);
        
    } catch (error) {
        showError(error.message || 'Failed to get recommendations');
    } finally {
        // Hide loading spinner
        loadingSpinner.classList.add('hidden');
    }
}

// Function to display recommendations
function displayRecommendations(data) {
    // Check if we have combined recommendations
    if (!data.songs || data.songs.length === 0) {
        showError('No recommendations found for the selected songs');
        return;
    }
    
    // Display combined recommendations
    recommendationsList.innerHTML = '';
    data.songs.forEach((song, index) => {
        const li = document.createElement('li');
        li.textContent = `${index + 1}. ${song}`;
        recommendationsList.appendChild(li);
    });
    
    // Display rule-based recommendations
    ruleRecommendationsList.innerHTML = '';
    if (data.rule_songs && data.rule_songs.length > 0) {
        data.rule_songs.forEach((song, index) => {
            const li = document.createElement('li');
            li.textContent = `${index + 1}. ${song}`;
            ruleRecommendationsList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No rule-based recommendations found.';
        li.classList.add('no-recommendations');
        ruleRecommendationsList.appendChild(li);
    }
    
    // Display similarity-based recommendations
    similarityRecommendationsList.innerHTML = '';
    if (data.similarity_songs && data.similarity_songs.length > 0) {
        data.similarity_songs.forEach((song, index) => {
            const li = document.createElement('li');
            li.textContent = `${index + 1}. ${song}`;
            similarityRecommendationsList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No similarity-based recommendations found.';
        li.classList.add('no-recommendations');
        similarityRecommendationsList.appendChild(li);
    }
    
    // Display model info
    if (data.version) {
        modelVersion.textContent = data.version;
        modelDate.textContent = data.model_date || 'Unknown';
        modelInfo.classList.remove('hidden');
    }
    
    // Scroll to results section
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

// Function to show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

// Function to hide error message
function hideError() {
    errorMessage.textContent = '';
    errorMessage.classList.add('hidden');
}

// Event Listeners
addSongBtn.addEventListener('click', addSong);

songInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        addSong();
    }
});

songList.addEventListener('click', removeSong);
clearSongsBtn.addEventListener('click', clearSongs);
getRecommendationsBtn.addEventListener('click', getRecommendations);

// Tab event listeners
tabButtons.forEach(button => {
    button.addEventListener('click', switchTab);
});

// Focus on input field when page loads
window.addEventListener('load', () => {
    songInput.focus();
}); 