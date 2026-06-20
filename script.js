/**
 * NeuroLex Frontend JavaScript v3.0
 * Enhanced with confidence tiers, chat, multi-language support, and robust error handling
 */

// ============================================================================
// CONFIGURATION & CONSTANTS
// ============================================================================

const CONFIG = {
    API_BASE: '',
    MIN_TEXT_LENGTH: 10,
    MAX_TEXT_LENGTH: 10000,
    DEBOUNCE_DELAY: 500,
    SUPPORTED_LANGUAGES: {
        'en': 'English',
        'ur': 'اردو (Urdu)',
        'ar': 'العربية (Arabic)',
        'ps': 'پښتو (Pashto)',
        'pa': 'ਪੰਜਾਬੀ (Punjabi)',
        'sd': 'سنڌي (Sindhi)',
        'bal': 'بلوچی (Balochi)'
    },
    CONFIDENCE_TIERS: {
        'HIGH': { color: '#10b981', label: 'High Confidence', icon: '✓' },
        'MEDIUM': { color: '#f59e0b', label: 'Medium Confidence', icon: '⚠' },
        'LOW': { color: '#ef4444', label: 'Low Confidence', icon: '!' },
        'UNCERTAIN': { color: '#6b7280', label: 'Uncertain', icon: '?' }
    }
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const STATE = {
    currentAnalysis: null,
    currentLanguage: 'en',
    chatHistory: [],
    chatSessionId: null,  // Session ID for conversation continuity
    modelLoaded: false,
    apiLimitReached: false,
    stats: {
        totalAnalyses: 0,
        fakeDetected: 0,
        realDetected: 0
    }
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 NeuroLex v3.0 initialized');
    initializeApp();
});

async function initializeApp() {
    try {
        // Check model status
        await checkModelStatus();
        
        // Attach event listeners
        attachEventListeners();
        
        // Initialize language selector
        initializeLanguageSelector();
        
        // Load statistics
        loadStatistics();
        
        console.log('✅ App initialization complete');
    } catch (error) {
        console.error('❌ Initialization error:', error);
        showNotification('Failed to initialize application', 'error');
    }
}

// ============================================================================
// MODEL STATUS & HEALTH CHECKS
// ============================================================================

async function checkModelStatus() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/readyz`);
        const data = await response.json();
        
        STATE.modelLoaded = data.model_loaded;
        
        // Update UI status indicator
        updateModelStatusUI(data.model_loaded);
        
        console.log('Model status:', data);
        return data.model_loaded;
    } catch (error) {
        console.error('Status check failed:', error);
        updateModelStatusUI(false);
        return false;
    }
}

function updateModelStatusUI(isLoaded) {
    const statusElements = document.querySelectorAll('.model-status');
    statusElements.forEach(el => {
        if (isLoaded) {
            el.textContent = '✓ Models Loaded';
            el.className = 'model-status loaded';
            el.style.color = '#10b981';
        } else {
            el.textContent = '⏳ Loading Models...';
            el.className = 'model-status loading';
            el.style.color = '#f59e0b';
        }
    });
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function attachEventListeners() {
    // Text analysis
    const analyzeBtn = document.querySelector('[onclick="analyzeTextMain()"]');
    if (analyzeBtn) {
        analyzeBtn.onclick = (e) => {
            e.preventDefault();
            analyzeTextMain();
        };
    }
    
    // Textarea keyboard shortcuts
    const textarea = document.getElementById('textareaInput');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                analyzeTextMain();
            }
        });
        
        // Character counter
        textarea.addEventListener('input', updateCharacterCount);
    }

    // Chat input keyboard shortcut (Enter to send)
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
    
    // URL analysis button
    const urlBtn = document.querySelector('[onclick="analyzeURL()"]');
    if (urlBtn) {
        urlBtn.onclick = (e) => {
            e.preventDefault();
            analyzeURL();
        };
    }
    
    // Image analysis button
    const imageBtn = document.querySelector('[onclick="analyzeImage()"]');
    if (imageBtn) {
        imageBtn.onclick = (e) => {
            e.preventDefault();
            analyzeImage();
        };
    }
    
    // Chat send button / form submission
    const chatForm = document.getElementById('chatForm');
    if (chatForm) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            sendChatMessage();
        });
    }
    
    const chatBtn = document.querySelector('[onclick="sendChatMessage()"]');
    if (chatBtn) {
        chatBtn.onclick = (e) => {
            e.preventDefault();
            sendChatMessage();
        };
    }
    
    // Language selector
    const langSelect = document.getElementById('languageSelect');
    if (langSelect) {
        langSelect.addEventListener('change', (e) => {
            STATE.currentLanguage = e.target.value;
            console.log('Language changed to:', STATE.currentLanguage);
        });
    }
    
    // Clear button
    const clearBtn = document.querySelector('[onclick="clearResults()"]');
    if (clearBtn) {
        clearBtn.onclick = (e) => {
            e.preventDefault();
            clearResults();
        };
    }
}

function updateCharacterCount() {
    const textarea = document.getElementById('textareaInput');
    const counter = document.getElementById('charCounter');
    
    if (textarea && counter) {
        const length = textarea.value.length;
        counter.textContent = `${length} / ${CONFIG.MAX_TEXT_LENGTH}`;
        
        if (length > CONFIG.MAX_TEXT_LENGTH) {
            counter.style.color = '#ef4444';
        } else if (length < CONFIG.MIN_TEXT_LENGTH) {
            counter.style.color = '#6b7280';
        } else {
            counter.style.color = '#10b981';
        }
    }
}

// ============================================================================
// LANGUAGE SELECTOR INITIALIZATION
// ============================================================================

function initializeLanguageSelector() {
    const langSelect = document.getElementById('languageSelect');
    if (!langSelect) return;
    
    // Clear existing options
    langSelect.innerHTML = '';
    
    // Add language options
    Object.entries(CONFIG.SUPPORTED_LANGUAGES).forEach(([code, name]) => {
        const option = document.createElement('option');
        option.value = code;
        option.textContent = name;
        if (code === STATE.currentLanguage) {
            option.selected = true;
        }
        langSelect.appendChild(option);
    });
}

// ============================================================================
// TEXT ANALYSIS - MAIN FUNCTION
// ============================================================================

async function analyzeTextMain() {
    const textarea = document.getElementById('textareaInput');
    const text = textarea ? textarea.value.trim() : '';
    
    // Validate input
    if (!text) {
        showNotification('Please enter some text to analyze', 'warning');
        return;
    }
    
    if (text.length < CONFIG.MIN_TEXT_LENGTH) {
        showNotification(`Text too short. Please enter at least ${CONFIG.MIN_TEXT_LENGTH} characters.`, 'warning');
        return;
    }
    
    if (text.length > CONFIG.MAX_TEXT_LENGTH) {
        showNotification(`Text too long. Maximum ${CONFIG.MAX_TEXT_LENGTH} characters allowed.`, 'warning');
        return;
    }
    
    // Check if model is loaded
    if (!STATE.modelLoaded) {
        const loaded = await checkModelStatus();
        if (!loaded) {
            showNotification('Models are still loading. Please wait...', 'info');
            return;
        }
    }
    
    // Show loading state
    showLoading('textResult', 'Analyzing text with AI models...');
    
    try {
        // Call the correct API endpoint
        const response = await fetch(`${CONFIG.API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                language: STATE.currentLanguage
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Store analysis
        STATE.currentAnalysis = result;
        
        // Update statistics
        updateStatistics(result);
        
        // Display result with enhanced UI
        displayTextResult(result);
        
        showNotification('Analysis complete!', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError('textResult', error.message || 'Failed to analyze text');
        showNotification('Analysis failed. Please try again.', 'error');
    }
}

// ============================================================================
// DISPLAY TEXT RESULT WITH CONFIDENCE TIERS
// ============================================================================

function displayTextResult(result) {
    const resultDiv = document.getElementById('textResult');
    if (!resultDiv) {
        console.error('Result div not found');
        return;
    }
    
    const isFake = result.label === 'FAKE';
    const confidence = (result.confidence * 100).toFixed(2);
    const realProb = (result.probabilities.REAL * 100).toFixed(2);
    const fakeProb = (result.probabilities.FAKE * 100).toFixed(2);
    
    // Get confidence tier info
    const tier = result.confidence_tier || 'MEDIUM';
    const tierInfo = CONFIG.CONFIDENCE_TIERS[tier];
    
    resultDiv.innerHTML = `
        <div class="result-card ${isFake ? 'fake-news' : 'real-news'}" style="animation: slideIn 0.3s ease-out;">
            <!-- Header -->
            <div class="result-header">
                <h3>
                    <span class="result-icon">${isFake ? '🚫' : '✅'}</span>
                    Analysis Result
                </h3>
                <span class="result-badge ${isFake ? 'badge-fake' : 'badge-real'}">
                    ${result.label}
                </span>
            </div>
            
            <!-- Confidence Tier Badge -->
            <div class="tier-badge" style="background-color: ${tierInfo.color}20; border-left: 4px solid ${tierInfo.color};">
                <span style="color: ${tierInfo.color};">${tierInfo.icon}</span>
                <strong>${tierInfo.label}</strong>
                <span style="font-size: 0.9em; color: #6b7280;">
                    (${tier === 'HIGH' ? '96%' : tier === 'MEDIUM' ? '82-88%' : tier === 'LOW' ? '70-80%' : '~60%'} accuracy expected)
                </span>
            </div>
            
            <!-- Main Confidence Score -->
            <div class="result-body">
                <div class="confidence-section">
                    <div class="confidence-label">Overall Confidence</div>
                    <div class="confidence-value" style="color: ${tierInfo.color};">${confidence}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${isFake ? 'fill-fake' : 'fill-real'}" 
                             style="width: ${confidence}%; background: linear-gradient(90deg, ${tierInfo.color}, ${isFake ? '#ef4444' : '#10b981'});"></div>
                    </div>
                </div>
                
                <!-- Probability Breakdown -->
                <div class="probabilities-section">
                    <h4>Detailed Probabilities</h4>
                    <div class="prob-item">
                        <span class="prob-label">
                            <span class="prob-icon">✅</span>
                            Real News
                        </span>
                        <span class="prob-value">${realProb}%</span>
                        <div class="prob-bar">
                            <div class="prob-fill fill-real" style="width: ${realProb}%"></div>
                        </div>
                    </div>
                    <div class="prob-item">
                        <span class="prob-label">
                            <span class="prob-icon">🚫</span>
                            Fake News
                        </span>
                        <span class="prob-value">${fakeProb}%</span>
                        <div class="prob-bar">
                            <div class="prob-fill fill-fake" style="width: ${fakeProb}%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Stage Breakdown (if available) -->
                ${result.stage_results ? displayStageResults(result.stage_results) : ''}
                
                <!-- Metadata -->
                <div class="result-meta">
                    <div class="meta-item">
                        <strong>Text Length:</strong> ${result.text_length} characters
                    </div>
                    <div class="meta-item">
                        <strong>Language:</strong> ${CONFIG.SUPPORTED_LANGUAGES[STATE.currentLanguage] || 'English'}
                    </div>
                    <div class="meta-item">
                        <strong>Analysis Time:</strong> ${new Date().toLocaleTimeString()}
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div class="result-actions">
                    <button onclick="getDetailedExplanation()" class="btn-secondary">
                        📝 Get Detailed Explanation
                    </button>
                    <button onclick="shareResult()" class="btn-secondary">
                        🔗 Share Result
                    </button>
                    <button onclick="reportIssue()" class="btn-secondary">
                        ⚠️ Report Issue
                    </button>
                </div>
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
    
    // Smooth scroll to result
    setTimeout(() => {
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// ============================================================================
// DISPLAY STAGE RESULTS
// ============================================================================

function displayStageResults(stageResults) {
    if (!stageResults || Object.keys(stageResults).length === 0) {
        return '';
    }
    
    let html = '<div class="stage-results"><h4>Pipeline Stage Analysis</h4>';
    
    const stageOrder = ['fact_check', 'domain', 'content', 'ensemble', 'pattern'];
    const stageNames = {
        'fact_check': '1. Fact Check',
        'domain': '2. Domain Analysis',
        'content': '3. Content Analysis',
        'ensemble': '4. AI Ensemble',
        'pattern': '5. Pattern Detection'
    };
    
    stageOrder.forEach(stage => {
        if (stageResults[stage]) {
            const data = stageResults[stage];
            html += `
                <div class="stage-item">
                    <div class="stage-name">${stageNames[stage]}</div>
                    <div class="stage-result ${data.label === 'FAKE' ? 'stage-fake' : 'stage-real'}">
                        ${data.label} (${(data.confidence * 100).toFixed(1)}%)
                    </div>
                </div>
            `;
        }
    });
    
    html += '</div>';
    return html;
}

// ============================================================================
// URL ANALYSIS
// ============================================================================

async function analyzeURL() {
    const urlInput = document.getElementById('urlInput');
    const url = urlInput ? urlInput.value.trim() : '';
    
    // Validate URL
    if (!url) {
        showNotification('Please enter a URL to analyze', 'warning');
        return;
    }
    
    if (!isValidURL(url)) {
        showNotification('Please enter a valid URL', 'warning');
        return;
    }
    
    showLoading('urlResult', 'Analyzing URL and extracting content...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/url/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                url: url,
                language: STATE.currentLanguage
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        displayURLResult(result);
        showNotification('URL analysis complete!', 'success');
        
    } catch (error) {
        console.error('URL analysis error:', error);
        showError('urlResult', error.message || 'Failed to analyze URL');
        showNotification('URL analysis failed', 'error');
    }
}

function displayURLResult(result) {
    const resultDiv = document.getElementById('urlResult');
    if (!resultDiv) return;
    
    const isFake = result.label === 'FAKE';
    const confidence = (result.confidence * 100).toFixed(2);
    const tier = result.confidence_tier || 'MEDIUM';
    const tierInfo = CONFIG.CONFIDENCE_TIERS[tier];
    
    resultDiv.innerHTML = `
        <div class="result-card ${isFake ? 'fake-news' : 'real-news'}">
            <div class="result-header">
                <h3>${isFake ? '🚫' : '✅'} URL Analysis Result</h3>
                <span class="result-badge ${isFake ? 'badge-fake' : 'badge-real'}">
                    ${result.label}
                </span>
            </div>
            
            <div class="tier-badge" style="background-color: ${tierInfo.color}20; border-left: 4px solid ${tierInfo.color};">
                <span style="color: ${tierInfo.color};">${tierInfo.icon}</span>
                <strong>${tierInfo.label}</strong>
            </div>
            
            <div class="result-body">
                <div class="confidence-section">
                    <div class="confidence-value" style="color: ${tierInfo.color};">${confidence}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${isFake ? 'fill-fake' : 'fill-real'}" 
                             style="width: ${confidence}%"></div>
                    </div>
                </div>
                
                ${result.domain_info ? `
                    <div class="domain-info">
                        <h4>Domain Information</h4>
                        <p><strong>Domain:</strong> ${result.domain_info.domain || 'Unknown'}</p>
                        <p><strong>Status:</strong> ${result.domain_info.status || 'Unknown'}</p>
                    </div>
                ` : ''}
                
                <div class="result-meta">
                    <small>URL: ${result.url || 'N/A'}</small>
                </div>
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

function isValidURL(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

// ============================================================================
// IMAGE ANALYSIS
// ============================================================================

async function analyzeImage() {
    const fileInput = document.getElementById('imageInput');
    
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        showNotification('Please select an image file', 'warning');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Please select a valid image file', 'warning');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showNotification('Image file too large. Maximum 10MB allowed.', 'warning');
        return;
    }
    
    showLoading('imageResult', 'Extracting text from image using OCR...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('language', STATE.currentLanguage);
        
        const response = await fetch(`${CONFIG.API_BASE}/api/image/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        displayImageResult(result);
        showNotification('Image analysis complete!', 'success');
        
    } catch (error) {
        console.error('Image analysis error:', error);
        showError('imageResult', error.message || 'Failed to analyze image');
        showNotification('Image analysis failed', 'error');
    }
}

function displayImageResult(result) {
    const resultDiv = document.getElementById('imageResult');
    if (!resultDiv) return;
    
    const isFake = result.label === 'FAKE';
    const confidence = (result.confidence * 100).toFixed(2);
    const tier = result.confidence_tier || 'MEDIUM';
    const tierInfo = CONFIG.CONFIDENCE_TIERS[tier];
    
    resultDiv.innerHTML = `
        <div class="result-card ${isFake ? 'fake-news' : 'real-news'}">
            <div class="result-header">
                <h3>${isFake ? '🚫' : '✅'} Image Analysis Result</h3>
                <span class="result-badge ${isFake ? 'badge-fake' : 'badge-real'}">
                    ${result.label}
                </span>
            </div>
            
            <div class="tier-badge" style="background-color: ${tierInfo.color}20; border-left: 4px solid ${tierInfo.color};">
                <span style="color: ${tierInfo.color};">${tierInfo.icon}</span>
                <strong>${tierInfo.label}</strong>
            </div>
            
            <div class="result-body">
                <div class="confidence-section">
                    <div class="confidence-value" style="color: ${tierInfo.color};">${confidence}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${isFake ? 'fill-fake' : 'fill-real'}" 
                             style="width: ${confidence}%"></div>
                    </div>
                </div>
                
                ${result.extracted_text ? `
                    <div class="extracted-text">
                        <h4>Extracted Text (OCR)</h4>
                        <div class="text-preview">
                            ${result.extracted_text.substring(0, 500)}
                            ${result.extracted_text.length > 500 ? '...' : ''}
                        </div>
                        <small>${result.extracted_text.length} characters extracted</small>
                    </div>
                ` : ''}
                
                <div class="result-meta">
                    <small>Image analyzed using multi-language OCR</small>
                </div>
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

// ============================================================================
// CHAT FUNCTIONALITY
// ============================================================================

async function sendChatMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput ? chatInput.value.trim() : '';
    
    if (!message) {
        showNotification('Please enter a message', 'warning');
        return;
    }
    
    // Add user message to chat
    addChatMessage('user', message);
    
    // Clear input
    chatInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Create or retrieve session ID for conversation continuity
        if (!STATE.chatSessionId) {
            STATE.chatSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        }
        
        const response = await fetch(`${CONFIG.API_BASE}/llm/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: STATE.chatSessionId,
                language: STATE.currentLanguage,
                include_context: true
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || 'Chat service unavailable');
        }
        
        const result = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add assistant response
        addChatMessage('assistant', result.response);
        
        // Update chat history
        STATE.chatHistory.push(
            { role: 'user', content: message },
            { role: 'assistant', content: result.response }
        );
        
        // Check if API limit reached
        if (result.fallback_mode) {
            STATE.apiLimitReached = true;
            showNotification('API limit reached. Using fallback mode.', 'info');
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        addChatMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
        showNotification('Chat failed. Please try again.', 'error');
    }
}

function addChatMessage(role, content) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}-message`;
    
    const avatar = role === 'user' ? '👤' : '🤖';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(content)}</div>
            <div class="message-time">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="typing-dots">
            <span></span><span></span><span></span>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// ============================================================================
// DETAILED EXPLANATION
// ============================================================================

async function getDetailedExplanation() {
    if (!STATE.currentAnalysis) {
        showNotification('No analysis available to explain', 'warning');
        return;
    }
    
    showLoading('explanationDiv', 'Generating detailed explanation...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/llm/explain`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: STATE.currentAnalysis.text || '',
                prediction: STATE.currentAnalysis.label,
                confidence: STATE.currentAnalysis.confidence,
                stage_results: STATE.currentAnalysis.stage_results || {}
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || 'Explanation service unavailable');
        }
        
        const result = await response.json();
        displayExplanation(result.explanation);
        
        showNotification('Explanation generated!', 'success');
        
    } catch (error) {
        console.error('Explanation error:', error);
        showError('explanationDiv', error.message || 'Failed to generate explanation');
        showNotification('Explanation failed', 'error');
    }
}

function displayExplanation(explanation) {
    const explanationDiv = document.getElementById('explanationDiv') || createExplanationDiv();
    
    explanationDiv.innerHTML = `
        <div class="explanation-card">
            <div class="explanation-header">
                <h3>📝 Detailed Explanation</h3>
                <button onclick="closeExplanation()" class="close-btn">×</button>
            </div>
            <div class="explanation-content">
                ${escapeHtml(explanation).replace(/\n/g, '<br>')}
            </div>
        </div>
    `;
    
    explanationDiv.style.display = 'block';
}

function createExplanationDiv() {
    const div = document.createElement('div');
    div.id = 'explanationDiv';
    div.className = 'explanation-container';
    document.body.appendChild(div);
    return div;
}

function closeExplanation() {
    const explanationDiv = document.getElementById('explanationDiv');
    if (explanationDiv) {
        explanationDiv.style.display = 'none';
    }
}

// ============================================================================
// STATISTICS & TRACKING
// ============================================================================

function updateStatistics(result) {
    STATE.stats.totalAnalyses++;
    
    if (result.label === 'FAKE') {
        STATE.stats.fakeDetected++;
    } else {
        STATE.stats.realDetected++;
    }
    
    // Update UI stats display
    updateStatsDisplay();
}

function updateStatsDisplay() {
    const statsElements = {
        total: document.getElementById('totalAnalyses'),
        fake: document.getElementById('fakeDetected'),
        real: document.getElementById('realDetected')
    };
    
    if (statsElements.total) {
        statsElements.total.textContent = STATE.stats.totalAnalyses;
    }
    if (statsElements.fake) {
        statsElements.fake.textContent = STATE.stats.fakeDetected;
    }
    if (statsElements.real) {
        statsElements.real.textContent = STATE.stats.realDetected;
    }
}

async function loadStatistics() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/stats/detailed`);
        if (response.ok) {
            const data = await response.json();
            console.log('Server statistics:', data);
        }
    } catch (error) {
        console.error('Failed to load statistics:', error);
    }
}

// ============================================================================
// LOADING & ERROR STATES
// ============================================================================

function showLoading(elementId, message = 'Processing...') {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.innerHTML = `
        <div class="loading-card">
            <div class="spinner-container">
                <div class="spinner"></div>
            </div>
            <p class="loading-message">${message}</p>
            <small class="loading-hint">This may take a few seconds...</small>
        </div>
    `;
    element.style.display = 'block';
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.innerHTML = `
        <div class="error-card">
            <div class="error-icon">⚠️</div>
            <h3>Error</h3>
            <p class="error-message">${escapeHtml(message)}</p>
            <button onclick="clearResults()" class="btn-secondary">
                Try Again
            </button>
        </div>
    `;
    element.style.display = 'block';
}

// ============================================================================
// NOTIFICATIONS
// ============================================================================

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelectorAll('.notification-toast');
    existing.forEach(el => el.remove());
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification-toast notification-${type}`;
    
    const icons = {
        'success': '✓',
        'error': '✕',
        'warning': '⚠',
        'info': 'ℹ'
    };
    
    notification.innerHTML = `
        <span class="notification-icon">${icons[type] || 'ℹ'}</span>
        <span class="notification-message">${escapeHtml(message)}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function clearResults() {
    const resultDivs = ['textResult', 'urlResult', 'imageResult'];
    
    resultDivs.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = '';
            element.style.display = 'none';
        }
    });
    
    // Clear inputs
    const textarea = document.getElementById('textareaInput');
    if (textarea) textarea.value = '';
    
    const urlInput = document.getElementById('urlInput');
    if (urlInput) urlInput.value = '';
    
    const imageInput = document.getElementById('imageInput');
    if (imageInput) imageInput.value = '';
    
    // Clear current analysis
    STATE.currentAnalysis = null;
    
    // Update character count
    updateCharacterCount();
    
    showNotification('Results cleared', 'info');
}

// ============================================================================
// SHARE FUNCTIONALITY
// ============================================================================

function shareResult() {
    if (!STATE.currentAnalysis) {
        showNotification('No analysis to share', 'warning');
        return;
    }
    
    const result = STATE.currentAnalysis;
    const shareText = `NeuroLex Analysis Result:
Label: ${result.label}
Confidence: ${(result.confidence * 100).toFixed(2)}%
Tier: ${result.confidence_tier}

Analyzed with NeuroLex v3.0 - AI-Powered Fake News Detection`;
    
    if (navigator.share) {
        navigator.share({
            title: 'NeuroLex Analysis Result',
            text: shareText
        }).then(() => {
            showNotification('Shared successfully!', 'success');
        }).catch(err => {
            console.error('Share failed:', err);
            copyToClipboard(shareText);
        });
    } else {
        copyToClipboard(shareText);
    }
}

function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Copy failed:', err);
            showNotification('Failed to copy', 'error');
        });
    } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            showNotification('Copied to clipboard!', 'success');
        } catch (err) {
            console.error('Copy failed:', err);
            showNotification('Failed to copy', 'error');
        }
        document.body.removeChild(textarea);
    }
}

// ============================================================================
// REPORT ISSUE
// ============================================================================

function reportIssue() {
    if (!STATE.currentAnalysis) {
        showNotification('No analysis to report', 'warning');
        return;
    }
    
    const modal = createReportModal();
    document.body.appendChild(modal);
}

function createReportModal() {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.id = 'reportModal';
    
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>Report Issue</h3>
                <button onclick="closeReportModal()" class="close-btn">×</button>
            </div>
            <div class="modal-body">
                <p>Help us improve by reporting incorrect predictions:</p>
                
                <label for="actualLabel">What should the correct label be?</label>
                <select id="actualLabel" class="form-control">
                    <option value="REAL">Real News</option>
                    <option value="FAKE">Fake News</option>
                </select>
                
                <label for="reportComments">Additional Comments (optional):</label>
                <textarea id="reportComments" class="form-control" rows="4" 
                          placeholder="Tell us why you think this prediction is incorrect..."></textarea>
                
                <div class="modal-footer">
                    <button onclick="closeReportModal()" class="btn-secondary">Cancel</button>
                    <button onclick="submitReport()" class="btn-primary">Submit Report</button>
                </div>
            </div>
        </div>
    `;
    
    return modal;
}

function closeReportModal() {
    const modal = document.getElementById('reportModal');
    if (modal) {
        modal.remove();
    }
}

async function submitReport() {
    const actualLabel = document.getElementById('actualLabel')?.value;
    const comments = document.getElementById('reportComments')?.value;
    
    if (!STATE.currentAnalysis) {
        showNotification('No analysis data available', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: STATE.currentAnalysis.text || '',
                actual_label: actualLabel,
                predicted_label: STATE.currentAnalysis.label,
                confidence: STATE.currentAnalysis.confidence,
                comments: comments
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to submit report');
        }
        
        closeReportModal();
        showNotification('Thank you for your feedback!', 'success');
        
    } catch (error) {
        console.error('Report submission error:', error);
        showNotification('Failed to submit report. Please try again.', 'error');
    }
}

// ============================================================================
// BATCH ANALYSIS
// ============================================================================

async function analyzeBatch() {
    const batchTextarea = document.getElementById('batchTextarea');
    if (!batchTextarea) {
        showNotification('Batch textarea not found', 'error');
        return;
    }
    
    const texts = batchTextarea.value.split('\n').filter(t => t.trim().length > 0);
    
    if (texts.length === 0) {
        showNotification('Please enter texts to analyze (one per line)', 'warning');
        return;
    }
    
    if (texts.length > 50) {
        showNotification('Maximum 50 texts allowed per batch', 'warning');
        return;
    }
    
    showLoading('batchResult', `Analyzing ${texts.length} texts...`);
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/batch/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                texts: texts,
                language: STATE.currentLanguage
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || 'Batch analysis failed');
        }
        
        const result = await response.json();
        displayBatchResults(result);
        showNotification(`Batch analysis complete! ${result.successful}/${result.total_processed} successful`, 'success');
        
    } catch (error) {
        console.error('Batch analysis error:', error);
        showError('batchResult', error.message || 'Failed to analyze batch');
        showNotification('Batch analysis failed', 'error');
    }
}

function displayBatchResults(result) {
    const resultDiv = document.getElementById('batchResult');
    if (!resultDiv) return;
    
    let html = `
        <div class="batch-results-card">
            <div class="batch-header">
                <h3>Batch Analysis Results</h3>
                <div class="batch-summary">
                    <span class="summary-item">Total: ${result.total_processed}</span>
                    <span class="summary-item">✓ Success: ${result.successful}</span>
                    <span class="summary-item">✕ Failed: ${result.failed}</span>
                </div>
            </div>
            
            <div class="batch-stats">
                <div class="stat-card fake-stat">
                    <div class="stat-value">${result.summary.fake_count}</div>
                    <div class="stat-label">Fake Detected</div>
                </div>
                <div class="stat-card real-stat">
                    <div class="stat-value">${result.summary.real_count}</div>
                    <div class="stat-label">Real Detected</div>
                </div>
                <div class="stat-card avg-stat">
                    <div class="stat-value">${(result.summary.average_confidence * 100).toFixed(1)}%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
            
            <div class="batch-items">
                <h4>Individual Results</h4>
    `;
    
    result.results.forEach((item, index) => {
        if (item.prediction) {
            const pred = item.prediction;
            const isFake = pred.label === 'FAKE';
            html += `
                <div class="batch-item ${isFake ? 'item-fake' : 'item-real'}">
                    <div class="item-number">#${index + 1}</div>
                    <div class="item-text">${escapeHtml(item.text)}</div>
                    <div class="item-result">
                        <span class="item-label ${isFake ? 'label-fake' : 'label-real'}">${pred.label}</span>
                        <span class="item-confidence">${(pred.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        } else if (item.error) {
            html += `
                <div class="batch-item item-error">
                    <div class="item-number">#${index + 1}</div>
                    <div class="item-text">${escapeHtml(item.text)}</div>
                    <div class="item-error-msg">Error: ${escapeHtml(item.error)}</div>
                </div>
            `;
        }
    });
    
    html += `
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
}

// ============================================================================
// LOADING & ERROR STATES
// ============================================================================

function showLoading(elementId, message = 'Processing...') {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.innerHTML = `
        <div class="loading-card">
            <div class="spinner-container">
                <div class="spinner"></div>
            </div>
            <p class="loading-message">${message}</p>
            <small class="loading-hint">This may take a few seconds...</small>
        </div>
    `;
    element.style.display = 'block';
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.innerHTML = `
        <div class="error-card">
            <div class="error-icon">⚠️</div>
            <h3>Error</h3>
            <p class="error-message">${escapeHtml(message)}</p>
            <button onclick="clearResults()" class="btn-secondary">
                Try Again
            </button>
        </div>
    `;
    element.style.display = 'block';
}

// ============================================================================
// NOTIFICATIONS
// ============================================================================

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelectorAll('.notification-toast');
    existing.forEach(el => el.remove());
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification-toast notification-${type}`;
    
    const icons = {
        'success': '✓',
        'error': '✕',
        'warning': '⚠',
        'info': 'ℹ'
    };
    
    notification.innerHTML = `
        <span class="notification-icon">${icons[type] || 'ℹ'}</span>
        <span class="notification-message">${escapeHtml(message)}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function clearResults() {
    const resultDivs = ['textResult', 'urlResult', 'imageResult'];
    
    resultDivs.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = '';
            element.style.display = 'none';
        }
    });
    
    // Clear inputs
    const textarea = document.getElementById('textareaInput');
    if (textarea) textarea.value = '';
    
    const urlInput = document.getElementById('urlInput');
    if (urlInput) urlInput.value = '';
    
    const imageInput = document.getElementById('imageInput');
    if (imageInput) imageInput.value = '';
    
    // Clear current analysis
    STATE.currentAnalysis = null;
    
    // Update character count
    updateCharacterCount();
    
    showNotification('Results cleared', 'info');
}

// ============================================================================
// SHARE FUNCTIONALITY
// ============================================================================

function shareResult() {
    if (!STATE.currentAnalysis) {
        showNotification('No analysis to share', 'warning');
        return;
    }
    
    const result = STATE.currentAnalysis;
    const shareText = `NeuroLex Analysis Result:
Label: ${result.label}
Confidence: ${(result.confidence * 100).toFixed(2)}%
Tier: ${result.confidence_tier}

Analyzed with NeuroLex v3.0 - AI-Powered Fake News Detection`;
    
    if (navigator.share) {
        navigator.share({
            title: 'NeuroLex Analysis Result',
            text: shareText
        }).then(() => {
            showNotification('Shared successfully!', 'success');
        }).catch(err => {
            console.error('Share failed:', err);
            copyToClipboard(shareText);
        });
    } else {
        copyToClipboard(shareText);
    }
}

function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Copy failed:', err);
            showNotification('Failed to copy', 'error');
        });
    } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            showNotification('Copied to clipboard!', 'success');
        } catch (err) {
            console.error('Copy failed:', err);
            showNotification('Failed to copy', 'error');
        }
        document.body.removeChild(textarea);
    }
}

// ============================================================================
// REPORT ISSUE
// ============================================================================

function reportIssue() {
    if (!STATE.currentAnalysis) {
        showNotification('No analysis to report', 'warning');
        return;
    }
    
    const modal = createReportModal();
    document.body.appendChild(modal);
}

function createReportModal() {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.id = 'reportModal';
    
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>Report Issue</h3>
                <button onclick="closeReportModal()" class="close-btn">×</button>
            </div>
            <div class="modal-body">
                <p>Help us improve by reporting incorrect predictions:</p>
                
                <label for="actualLabel">What should the correct label be?</label>
                <select id="actualLabel" class="form-control">
                    <option value="REAL">Real News</option>
                    <option value="FAKE">Fake News</option>
                </select>
                
                <label for="reportComments">Additional Comments (optional):</label>
                <textarea id="reportComments" class="form-control" rows="4" 
                          placeholder="Tell us why you think this prediction is incorrect..."></textarea>
                
                <div class="modal-footer">
                    <button onclick="closeReportModal()" class="btn-secondary">Cancel</button>
                    <button onclick="submitReport()" class="btn-primary">Submit Report</button>
                </div>
            </div>
        </div>
    `;
    
    return modal;
}

function closeReportModal() {
    const modal = document.getElementById('reportModal');
    if (modal) {
        modal.remove();
    }
}

async function submitReport() {
    const actualLabel = document.getElementById('actualLabel')?.value;
    const comments = document.getElementById('reportComments')?.value;
    
    if (!STATE.currentAnalysis) {
        showNotification('No analysis data available', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: STATE.currentAnalysis.text || '',
                actual_label: actualLabel,
                predicted_label: STATE.currentAnalysis.label,
                confidence: STATE.currentAnalysis.confidence,
                comments: comments
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to submit report');
        }
        
        closeReportModal();
        showNotification('Thank you for your feedback!', 'success');
        
    } catch (error) {
        console.error('Report submission error:', error);
        showNotification('Failed to submit report. Please try again.', 'error');
    }
}

// ============================================================================
// BATCH ANALYSIS
// ============================================================================

async function analyzeBatch() {
    const batchTextarea = document.getElementById('batchTextarea');
    if (!batchTextarea) {
        showNotification('Batch textarea not found', 'error');
        return;
    }
    
    const texts = batchTextarea.value.split('\n').filter(t => t.trim().length > 0);
    
    if (texts.length === 0) {
        showNotification('Please enter texts to analyze (one per line)', 'warning');
        return;
    }
    
    if (texts.length > 50) {
        showNotification('Maximum 50 texts allowed per batch', 'warning');
        return;
    }
    
    showLoading('batchResult', `Analyzing ${texts.length} texts...`);
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/batch/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                texts: texts,
                language: STATE.currentLanguage
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || 'Batch analysis failed');
        }
        
        const result = await response.json();
        displayBatchResults(result);
        showNotification(`Batch analysis complete! ${result.successful}/${result.total_processed} successful`, 'success');
        
    } catch (error) {
        console.error('Batch analysis error:', error);
        showError('batchResult', error.message || 'Failed to analyze batch');
        showNotification('Batch analysis failed', 'error');
    }
}

function displayBatchResults(result) {
    const resultDiv = document.getElementById('batchResult');
    if (!resultDiv) return;
    
    let html = `
        <div class="batch-results-card">
            <div class="batch-header">
                <h3>Batch Analysis Results</h3>
                <div class="batch-summary">
                    <span class="summary-item">Total: ${result.total_processed}</span>
                    <span class="summary-item">✓ Success: ${result.successful}</span>
                    <span class="summary-item">✕ Failed: ${result.failed}</span>
                </div>
            </div>
            
            <div class="batch-stats">
                <div class="stat-card fake-stat">
                    <div class="stat-value">${result.summary.fake_count}</div>
                    <div class="stat-label">Fake Detected</div>
                </div>
                <div class="stat-card real-stat">
                    <div class="stat-value">${result.summary.real_count}</div>
                    <div class="stat-label">Real Detected</div>
                </div>
                <div class="stat-card avg-stat">
                    <div class="stat-value">${(result.summary.average_confidence * 100).toFixed(1)}%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
            
            <div class="batch-items">
                <h4>Individual Results</h4>
    `;
    
    result.results.forEach((item, index) => {
        if (item.prediction) {
            const pred = item.prediction;
            const isFake = pred.label === 'FAKE';
            html += `
                <div class="batch-item ${isFake ? 'item-fake' : 'item-real'}">
                    <div class="item-number">#${index + 1}</div>
                    <div class="item-text">${escapeHtml(item.text)}</div>
                    <div class="item-result">
                        <span class="item-label ${isFake ? 'label-fake' : 'label-real'}">${pred.label}</span>
                        <span class="item-confidence">${(pred.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        } else if (item.error) {
            html += `
                <div class="batch-item item-error">
                    <div class="item-number">#${index + 1}</div>
                    <div class="item-text">${escapeHtml(item.text)}</div>
                    <div class="item-error-msg">Error: ${escapeHtml(item.error)}</div>
                </div>
            `;
        }
    });
    
    html += `
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
}

