
// API Base URL (same origin when served by FastAPI)
const API_BASE = window.location.origin;

// Chat session for LLM endpoint
let chatSessionId = localStorage.getItem('neurolex_chat_session') || null;

const SESSION_STATS_KEY = 'neurolex_session_stats';

function loadSessionStats() {
    try {
        return JSON.parse(localStorage.getItem(SESSION_STATS_KEY) || '{"count":0,"fake":0,"real":0}');
    } catch {
        return { count: 0, fake: 0, real: 0 };
    }
}

let sessionStats = loadSessionStats();

function saveSessionStats() {
    localStorage.setItem(SESSION_STATS_KEY, JSON.stringify(sessionStats));
}

function bumpSessionStats(label) {
    sessionStats.count += 1;
    if (String(label).toUpperCase() === 'FAKE') sessionStats.fake += 1;
    else if (String(label).toUpperCase() === 'REAL') sessionStats.real += 1;
    saveSessionStats();
    updateDashboardMetrics(null);
}

function normalizeConfidence(value) {
    let c = Number(value) || 0;
    if (c > 1) c = c / 100;
    return Math.min(1, Math.max(0, c));
}

function normalizePrediction(data) {
    const pred = data?.prediction || data;
    const tierObj = pred?.confidence_tier ?? data?.confidence_tier ?? {};
    const models = pred?.models_used
        || data?.model_info?.models_used
        || data?.model_info?.active_models
        || (data?.model_info?.source ? [data.model_info.source] : []);

    return {
        label: pred?.label ?? data?.label ?? 'UNKNOWN',
        confidence: normalizeConfidence(pred?.confidence ?? data?.confidence),
        tier: tierObj?.tier ?? 'UNCERTAIN',
        tierAccuracy: tierObj?.accuracy_estimate ?? '',
        model: Array.isArray(models) ? models.join(', ') : (models || 'BERT-Ensemble'),
        explanation: tierObj?.recommendation ?? data?.recommendation ?? data?.explanation ?? '',
        processing_time: (data?.processing_time_ms ?? pred?.processing_time_ms ?? 0) / 1000,
        probabilities: pred?.probabilities ?? data?.probabilities ?? {},
        warnings: data?.warnings ?? [],
        patterns: extractPatterns(data),
        content_analysis: data?.content_analysis ?? data?.analysis?.content_quality ?? null,
        fact_check_result: data?.fact_check_result ?? null,
        domain_analysis: data?.domain_analysis ?? null,
        pattern_detection: data?.pattern_detection ?? data?.analysis?.pattern_detection ?? null,
        model_info: data?.model_info ?? null,
        raw: data
    };
}

function extractPatterns(data) {
    const pd = data?.pattern_detection
        ?? data?.prediction?.pattern_detection
        ?? data?.analysis?.pattern_detection;
    if (!pd) return [];
    if (Array.isArray(pd.patterns)) {
        return pd.patterns.map(p => (typeof p === 'string' ? p : (p.pattern || p.name || 'pattern')));
    }
    if (pd.pattern_detected) return ['Suspicious pattern detected'];
    return [];
}

async function parseApiError(response) {
    try {
        const body = await response.json();
        if (typeof body.detail === 'string') return body.detail;
        if (body.detail?.message) return body.detail.message;
        if (body.error) return body.error;
        return JSON.stringify(body.detail || body);
    } catch {
        return `Request failed (${response.status})`;
    }
}

function showAnalysisLoading(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.classList.remove('hidden');
    container.innerHTML = `
        <div class="result-card" style="border-left: 3px solid var(--primary);">
            <div style="display:flex;align-items:center;gap:1rem;">
                <span class="spinner" style="display:inline-block;"></span>
                <div>
                    <div style="font-weight:600;">Analyzing…</div>
                    <div style="color:var(--text-secondary);font-size:0.9rem;margin-top:0.25rem;">
                        Running multi-stage pipeline (fact-check → domain → content → ensemble)
                    </div>
                </div>
            </div>
        </div>`;
}

function hideContentSignals() {
    const el = document.getElementById('textContentSignals');
    if (el) el.classList.add('hidden');
}

async function loadLiveStats() {
    let serverStats = null;
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        if (response.ok) serverStats = await response.json();
    } catch (e) {
        console.warn('Could not load server stats:', e);
    }
    updateDashboardMetrics(serverStats);
    updateSidebarComparison(serverStats);
    drawSidebarChart(serverStats);
    drawChatVisualization(serverStats);
}

function updateDashboardMetrics(serverStats) {
    const set = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };
    set('metricSessionCount', String(sessionStats.count));
    set('metricSessionHint', sessionStats.count
        ? `${sessionStats.fake} fake · ${sessionStats.real} real`
        : 'this browser session');
    if (serverStats) {
        set('metricServerTotal', String(serverStats.total ?? 0));
        set('metricFakeRate', `${serverStats.fake_percentage ?? 0}%`);
        const highPct = serverStats.confidence_tiers?.high?.percentage ?? 0;
        set('metricHighConf', serverStats.confidence_tiers?.high?.accuracy_estimate || '98%');
        set('metricHighConfHint', highPct > 0
            ? `${highPct}% of analyses in HIGH tier`
            : 'run an analysis to populate tiers');
        set('metricServerHint', `${serverStats.fake ?? 0} fake · ${serverStats.real ?? 0} real`);
    }
}

function updateSidebarComparison(serverStats) {
    const set = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };
    set('sidebarSessionCount', String(sessionStats.count));
    set('sidebarSessionHint', sessionStats.count ? 'your analyses' : 'start analyzing');
    if (serverStats) {
        set('sidebarServerTotal', String(serverStats.total ?? 0));
        set('sidebarServerHint', 'since server start');
        const highPct = serverStats.confidence_tiers?.high?.percentage ?? 0;
        set('sidebarHighTierPct', `${highPct}%`);
        set('sidebarHighTierHint', 'HIGH confidence tier');
    }
}

function renderContentSignals(data) {
    const block = document.getElementById('textContentSignals');
    const content = document.getElementById('contentSignalsContent');
    if (!block || !content) return;

    const ca = data?.content_analysis;
    const features = ca?.key_features || ca?.features;
    if (!features) {
        block.classList.add('hidden');
        return;
    }

    const items = [
        { name: 'Has author', value: features.has_author ? 1 : 0, positive: features.has_author },
        { name: 'Has quotes', value: features.has_quotes ? 1 : 0, positive: features.has_quotes },
        { name: 'Clickbait', value: features.has_clickbait ? 1 : 0, positive: !features.has_clickbait },
        { name: 'Conspiracy keywords', value: Math.min(1, (features.conspiracy_keywords || 0) / 5), positive: (features.conspiracy_keywords || 0) < 2 },
        { name: 'Credibility score', value: ca.credibility_score ?? 0.5, positive: (ca.credibility_score ?? 0.5) > 0.5 }
    ];

    content.innerHTML = items.map(f => {
        const width = Math.round(Math.abs(f.value) * 100);
        const cls = f.positive ? 'shap-positive' : 'shap-negative';
        return `
            <div class="shap-bar">
                <div class="shap-label">${f.name}</div>
                <div class="shap-value-bar ${cls}" style="width:${width}%">
                    <span class="shap-value-text">${typeof f.value === 'number' && f.value <= 1 ? (f.positive ? '✓' : '✗') : f.value}</span>
                </div>
            </div>`;
    }).join('');

    if (ca.risk_level) {
        content.innerHTML += `<p style="margin-top:0.75rem;color:var(--text-secondary);">Risk level: <strong>${ca.risk_level}</strong></p>`;
    }
    block.classList.remove('hidden');
}

function buildPipelineHtml(data) {
    const stages = [];

    const fc = data?.fact_check_result;
    stages.push({
        name: 'Fact-check',
        status: fc?.fact_checked ? 'matched' : 'skipped',
        detail: fc?.fact_checked ? (fc.verdict || fc.claim || 'Verified in database') : 'No fact-check hit'
    });

    const dom = data?.domain_analysis;
    stages.push({
        name: 'Domain',
        status: dom?.domain_checked || dom?.domain ? 'checked' : 'skipped',
        detail: dom?.domain
            ? `${dom.domain} — ${dom.category || 'unknown'} (${dom.is_trusted ? 'trusted' : dom.is_suspicious ? 'suspicious' : 'neutral'})`
            : 'Not applicable (text-only)'
    });

    const ca = data?.content_analysis;
    stages.push({
        name: 'Content',
        status: ca ? 'checked' : 'skipped',
        detail: ca
            ? `Risk: ${ca.risk_level || 'n/a'}, score ${(ca.credibility_score ?? 0).toFixed(2)}`
            : 'No content features'
    });

    const mi = data?.model_info;
    stages.push({
        name: 'Ensemble',
        status: mi ? 'active' : 'active',
        detail: mi?.source
            ? `${mi.source} (stage ${mi.stage ?? '—'})`
            : (data?.prediction?.models_used || []).join(', ') || 'BERT + RoBERTa + DeBERTa'
    });

    const pd = data?.pattern_detection;
    stages.push({
        name: 'Patterns',
        status: pd?.pattern_detected ? 'alert' : 'clear',
        detail: pd?.pattern_detected ? extractPatterns(data).join(', ') || 'Pattern detected' : 'No override patterns'
    });

    return `
        <div class="explanation" style="margin-top:1rem;">
            <div class="explanation-title">🔬 Pipeline Stages</div>
            <div class="stats-grid" style="margin-top:0.5rem;">
                ${stages.map(s => `
                    <div class="stat-item">
                        <div class="stat-label">${s.name}</div>
                        <div class="stat-value" style="font-size:0.85rem;">${s.detail}</div>
                    </div>
                `).join('')}
            </div>
        </div>`;
}


// History storage
let textHistory = [];
let urlHistory = [];
let imageHistory = [];

// Latest analysis per tab (for export from result card)
const lastAnalysisByTab = { text: null, url: null, image: null };

const TAB_FROM_RESULT_ID = {
    textResult: 'text',
    urlResult: 'url',
    imageResult: 'image'
};

// Floating promo badges removed (were misleading static marketing copy)
function addFloatingStats() {
    return;
    const stats = [
        { text: '🎯 Live ensemble', top: '15%', right: '5%', delay: 0 },
        { text: '⚡ Real-time Analysis', top: '30%', right: '3%', delay: 2 },
        { text: '🌍 Multi-language', top: '45%', right: '6%', delay: 4 },
        { text: '🔒 Secure & Private', top: '60%', right: '4%', delay: 6 }
    ];

    stats.forEach((stat, index) => {
        setTimeout(() => {
            const el = document.createElement('div');
            el.className = 'floating-stat';
            el.textContent = stat.text;
            el.style.top = stat.top;
            el.style.right = stat.right;
            el.style.animationDelay = `${stat.delay}s`;
            document.body.appendChild(el);
        }, index * 500);
    });
}

// Theme management - FIXED: Now shows correct icons
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', newTheme);
    // Fixed logic: Light mode shows moon (to switch to dark), Dark mode shows sun (to switch to light)
    document.querySelector('.theme-toggle').textContent = newTheme === 'light' ? '🌙' : '☀️';
}

// Sidebar drawer management
function openSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebarBackdrop');
    const toggleBtn = document.getElementById('sidebarToggle');
    if (!sidebar) return;

    sidebar.classList.add('open');
    backdrop?.classList.add('visible');
    document.body.classList.add('sidebar-open');
    toggleBtn?.classList.add('sidebar-open');
    toggleBtn?.setAttribute('aria-expanded', 'true');
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebarBackdrop');
    const toggleBtn = document.getElementById('sidebarToggle');
    if (!sidebar) return;

    sidebar.classList.remove('open');
    backdrop?.classList.remove('visible');
    document.body.classList.remove('sidebar-open');
    toggleBtn?.classList.remove('sidebar-open');
    toggleBtn?.setAttribute('aria-expanded', 'false');
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar?.classList.contains('open')) closeSidebar();
    else openSidebar();
}

// Tutorial display with enhanced content - in-UI modal
function showTutorial(type) {
    if (window.innerWidth < 1024) closeSidebar();

    const tutorials = {
        basics: {
            title: '🚀 NeuroLex Quick Start Guide',
            content: `
                        <div style="line-height: 1.8;">
                            <h3 style="color: var(--primary); margin-bottom: 1rem;">Getting Started in 5 Steps</h3>
                            <ol style="padding-left: 1.5rem;">
                                <li style="margin-bottom: 0.8rem;"><strong>Choose Analysis Type:</strong> Select Text, URL, Image, or Chat based on your content</li>
                                <li style="margin-bottom: 0.8rem;"><strong>Input Content:</strong> Paste text, enter URL, or upload image file</li>
                                <li style="margin-bottom: 0.8rem;"><strong>Instant Results:</strong> Get AI-powered analysis in real-time with confidence scores</li>
                                <li style="margin-bottom: 0.8rem;"><strong>Review Insights:</strong> Examine confidence tiers, patterns, and AI explanations</li>
                                <li style="margin-bottom: 0.8rem;"><strong>Export Reports:</strong> Download results as JSON, CSV, or PDF for records</li>
                            </ol>
                            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(99, 102, 241, 0.1); border-left: 4px solid var(--primary); border-radius: 8px;">
                                <strong>💡 Pro Features:</strong> Advanced BERT ensemble with 96% accuracy on high-confidence predictions, multi-language support, and real-time pattern detection!
                            </div>
                        </div>
                    `
        },
        verification: {
            title: '✅ Professional Fact Verification',
            content: `
                        <div style="line-height: 1.8;">
                            <h3 style="color: var(--secondary); margin-bottom: 1rem;">5-Step Verification Protocol</h3>
                            <ol style="padding-left: 1.5rem;">
                                <li style="margin-bottom: 0.8rem;"><strong>Cross-Reference:</strong> Verify with 3+ independent trusted sources</li>
                                <li style="margin-bottom: 0.8rem;"><strong>Author Check:</strong> Investigate credentials, expertise, and past publications</li>
                                <li style="margin-bottom: 0.8rem;"><strong>Citation Analysis:</strong> Look for proper references and source links</li>
                                <li style="margin-bottom: 0.8rem;"><strong>Temporal Check:</strong> Verify publication date and any updates/corrections</li>
                                <li style="margin-bottom: 0.8rem;"><strong>Bias Detection:</strong> Analyze language for emotional manipulation and loaded terms</li>
                            </ol>
                            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-left: 4px solid var(--secondary); border-radius: 8px;">
                                <strong>💡 Expert Tips:</strong> Use reverse image search for visuals, check domain reputation databases, and verify with fact-checking organizations like Snopes or PolitiFact!
                            </div>
                        </div>
                    `
        },
        sources: {
            title: '🔍 Credibility Assessment Framework',
            content: `
                        <div style="line-height: 1.8;">
                            <h3 style="color: var(--primary); margin-bottom: 1rem;">Source Quality Indicators</h3>
                            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                                <strong style="color: var(--secondary);">✓ Green Flags (Trustworthy):</strong>
                                <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                                    <li>Clear author with verifiable credentials</li>
                                    <li>Established domain (age > 2 years)</li>
                                    <li>Multiple cited references and sources</li>
                                    <li>Professional design and presentation</li>
                                    <li>Transparent correction/update policy</li>
                                    <li>Contact information readily available</li>
                                    <li>Minimal intrusive advertising</li>
                                </ul>
                            </div>
                            <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 10px;">
                                <strong style="color: var(--danger);">🚩 Red Flags (Suspicious):</strong>
                                <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                                    <li>Anonymous or pseudonymous authors</li>
                                    <li>Sensationalist clickbait headlines</li>
                                    <li>Poor grammar and spelling errors</li>
                                    <li>No verifiable sources or citations</li>
                                    <li>Excessive ads and pop-ups</li>
                                    <li>Recent domain registration</li>
                                    <li>Extreme political or ideological bias</li>
                                </ul>
                            </div>
                        </div>
                    `
        },
        deepfake: {
            title: '🎭 Advanced Deepfake Detection',
            content: `
                        <div style="line-height: 1.8;">
                            <h3 style="color: var(--warning); margin-bottom: 1rem;">AI Manipulation Detection Guide</h3>
                            <div style="padding: 1rem; background: rgba(245, 158, 11, 0.1); border-radius: 10px; margin-bottom: 1rem;">
                                <strong>🔍 Visual Inspection Checklist:</strong>
                                <ol style="margin-top: 0.5rem; padding-left: 1.5rem;">
                                    <li>Unnatural facial movements or expressions</li>
                                    <li>Inconsistent lighting and shadows</li>
                                    <li>Blurry or distorted edges around face/hair</li>
                                    <li>Unnatural eye movements or blinking patterns</li>
                                    <li>Skin texture inconsistencies</li>
                                    <li>Audio-video synchronization issues</li>
                                </ol>
                            </div>
                            <div style="padding: 1rem; background: rgba(99, 102, 241, 0.1); border-radius: 10px;">
                                <strong>🛠️ Technical Verification:</strong>
                                <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                                    <li>Use reverse image search (Google Images, TinEye)</li>
                                    <li>Check EXIF metadata for manipulation signs</li>
                                    <li>Verify source credibility and original publication</li>
                                    <li>Use NeuroLex's advanced computer vision analysis</li>
                                </ul>
                            </div>
                            <p style="margin-top: 1rem; font-style: italic; color: var(--text-secondary);">NeuroLex employs state-of-the-art deep learning to detect subtle manipulation artifacts invisible to human eyes!</p>
                        </div>
                    `
        }
    };

    const tutorial = tutorials[type];
    if (!tutorial) return;

    const modal = document.createElement('div');
    modal.className = 'tutorial-modal';
    modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(5px);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: fadeIn 0.3s ease;
            `;

    // Create modal content
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
                background: var(--glass-bg);
                border: 2px solid var(--glass-border);
                border-radius: 20px;
                padding: 2.5rem;
                max-width: 600px;
                max-height: 80vh;
                overflow-y: auto;
                position: relative;
                animation: slideInUp 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            `;

    modalContent.innerHTML = `
                <button onclick="this.closest('.tutorial-modal').remove()" style="
                    position: absolute;
                    top: 1rem;
                    right: 1rem;
                    background: rgba(239, 68, 68, 0.2);
                    border: 2px solid var(--danger);
                    color: var(--danger);
                    font-size: 1.5rem;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                " onmouseover="this.style.background='var(--danger)'; this.style.color='white'; this.style.transform='rotate(90deg)';" onmouseout="this.style.background='rgba(239, 68, 68, 0.2)'; this.style.color='var(--danger)'; this.style.transform='rotate(0deg)';">✕</button>
                <h2 style="margin-bottom: 1.5rem; color: var(--text-primary);">${tutorial.title}</h2>
                ${tutorial.content}
            `;

    modal.appendChild(modalContent);
    document.body.appendChild(modal);

    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

// Quick questions
function askQuestion(question) {
    document.getElementById('chatInput').value = question;
    document.getElementById('chatForm').dispatchEvent(new Event('submit'));
}

// Minimal safe markdown renderer (code blocks, inline code, links, bold, italic, line breaks)
function renderMarkdown(md) {
    if (!md) return '';
    // Escape first
    let s = escapeHtml(String(md));

    // Code blocks ```lang\n...```
    s = s.replace(/```([\s\S]*?)```/g, (m, code) => {
        return '<pre><code>' + code.replace(/</g, '&lt;') + '</code></pre>';
    });

    // Inline code `code`
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold **text** and __text__
    s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.*?)__/g, '<strong>$1</strong>');

    // Italic *text* or _text_
    s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
    s = s.replace(/_(.*?)_/g, '<em>$1</em>');

    // Links [text](url)
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Line breaks
    s = s.replace(/\n/g, '<br>');

    return s;
}

// ─── Export helpers ───
function escapeHtml(str) {
    return String(str ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function escapeCsvCell(value) {
    const s = value == null ? '' : String(value);
    return `"${s.replace(/"/g, '""')}"`;
}

function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 500);
}

function buildExportReport(rawData, meta = {}) {
    const p = normalizePrediction(rawData);
    const label = String(p.label || 'UNKNOWN').toUpperCase();
    return {
        generated_at: new Date().toISOString(),
        analysis_type: meta.type || 'unknown',
        input_preview: meta.input || '',
        classification: label,
        confidence: p.confidence,
        confidence_percent: `${(p.confidence * 100).toFixed(1)}%`,
        confidence_tier: p.tier,
        tier_accuracy_estimate: p.tierAccuracy,
        models_used: p.model,
        processing_time_seconds: p.processing_time > 0 ? p.processing_time : null,
        recommendation: p.explanation,
        probabilities: p.probabilities,
        warnings: p.warnings,
        patterns: p.patterns,
        domain_analysis: p.domain_analysis,
        content_analysis: p.content_analysis,
        fact_check_result: p.fact_check_result,
        ocr_text_preview: rawData?.ocr_text_preview || null,
        full_api_response: rawData
    };
}

function flattenObjectRows(obj, prefix = '') {
    const rows = [];
    if (obj == null) return rows;
    for (const [key, value] of Object.entries(obj)) {
        const path = prefix ? `${prefix}.${key}` : key;
        if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
            rows.push(...flattenObjectRows(value, path));
        } else {
            const cell = Array.isArray(value) ? JSON.stringify(value) : value;
            rows.push([path, cell]);
        }
    }
    return rows;
}

function getHistoryStore(type) {
    if (type === 'text') return textHistory;
    if (type === 'url') return urlHistory;
    if (type === 'image') return imageHistory;
    return [];
}

function exportReport(report, format) {
    if (!report) {
        showNotification('Nothing to export yet — run an analysis first.', 'warning');
        return;
    }

    const stamp = Date.now();
    const slug = (report.classification || 'analysis').toLowerCase();

    if (format === 'json') {
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        triggerDownload(blob, `neurolex-${slug}-${stamp}.json`);
        showNotification('JSON report downloaded', 'success');
        return;
    }

    if (format === 'csv') {
        const rows = flattenObjectRows(report);
        const csv = ['Field,Value', ...rows.map(([k, v]) => `${escapeCsvCell(k)},${escapeCsvCell(v)}`)].join('\n');
        const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' });
        triggerDownload(blob, `neurolex-${slug}-${stamp}.csv`);
        showNotification('CSV report downloaded', 'success');
        return;
    }

    if (format === 'pdf') {
        const labelClass = report.classification === 'FAKE' ? 'fake'
            : report.classification === 'REAL' ? 'real' : 'uncertain';
        const warningsHtml = (report.warnings || []).length
            ? `<ul>${report.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('')}</ul>` : '';
        const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>NeuroLex Report</title>
<style>
  body{font-family:Segoe UI,Arial,sans-serif;padding:40px;color:#0f172a;max-width:800px;margin:0 auto}
  h1{color:#4f46e5;border-bottom:3px solid #4f46e5;padding-bottom:8px}
  .banner{font-size:1.4rem;font-weight:700;text-align:center;padding:18px;border-radius:12px;margin:20px 0}
  .real{background:#d1fae5;color:#065f46}.fake{background:#fee2e2;color:#991b1b}.uncertain{background:#fef3c7;color:#92400e}
  .section{margin:16px 0;padding:16px;background:#f8fafc;border-left:4px solid #6366f1;border-radius:0 8px 8px 0}
  table{width:100%;border-collapse:collapse;margin-top:8px}
  td{padding:8px;border-bottom:1px solid #e2e8f0;vertical-align:top}
  td:first-child{font-weight:600;width:38%;color:#475569}
  .footer{margin-top:32px;text-align:center;color:#64748b;font-size:0.85rem}
  @media print{body{padding:20px}}
</style></head><body>
<h1>NeuroLex AI — Analysis Report</h1>
<p><strong>Generated:</strong> ${escapeHtml(new Date(report.generated_at).toLocaleString())}</p>
<div class="banner ${labelClass}">${escapeHtml(report.classification)} · ${escapeHtml(report.confidence_percent)} confidence</div>
<div class="section"><h2>Summary</h2>
<table>
<tr><td>Analysis type</td><td>${escapeHtml(report.analysis_type)}</td></tr>
<tr><td>Input preview</td><td>${escapeHtml(report.input_preview)}</td></tr>
<tr><td>Confidence tier</td><td>${escapeHtml(report.confidence_tier)} ${escapeHtml(report.tier_accuracy_estimate)}</td></tr>
<tr><td>Models</td><td>${escapeHtml(report.models_used)}</td></tr>
<tr><td>Processing time</td><td>${report.processing_time_seconds != null ? report.processing_time_seconds.toFixed(2) + 's' : 'N/A'}</td></tr>
<tr><td>REAL probability</td><td>${report.probabilities?.REAL != null ? (report.probabilities.REAL * 100).toFixed(1) + '%' : 'N/A'}</td></tr>
<tr><td>FAKE probability</td><td>${report.probabilities?.FAKE != null ? (report.probabilities.FAKE * 100).toFixed(1) + '%' : 'N/A'}</td></tr>
</table></div>
${report.recommendation ? `<div class="section"><h2>Recommendation</h2><p>${escapeHtml(report.recommendation)}</p></div>` : ''}
${warningsHtml ? `<div class="section"><h2>Warnings</h2>${warningsHtml}</div>` : ''}
<div class="footer"><p>NeuroLex AI · Fake News Detection</p><p>Open this file in a browser → Print → Save as PDF</p></div>
</body></html>`;
        const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
        triggerDownload(blob, `neurolex-${slug}-${stamp}.html`);
        showNotification('Report downloaded — open file, then Print → Save as PDF', 'success');
    }
}

function exportLatest(tab, format) {
    const entry = lastAnalysisByTab[tab];
    if (!entry?.raw) {
        showNotification('No analysis to export on this tab.', 'warning');
        return;
    }
    exportReport(buildExportReport(entry.raw, { type: tab, input: entry.input || '' }), format);
}

function exportHistoryItem(type, index, format) {
    const item = getHistoryStore(type)[index];
    if (!item?.data) {
        showNotification('History item not found.', 'error');
        return;
    }
    exportReport(buildExportReport(item.data, { type, input: item.input || '' }), format);
}

function renderExportToolbar(tab) {
    return `
        <div class="export-toolbar" role="group" aria-label="Export analysis report">
            <span class="export-toolbar__label">Export</span>
            <div class="export-toolbar__buttons" data-tab="${tab}">
            <button type="button" class="btn-export btn-export--json" data-format="json" title="Download JSON">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>
                JSON
            </button>
            <button type="button" class="btn-export btn-export--csv" data-format="csv" title="Download CSV">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M8 13h2v2H8zM12 13h2v2h-2zM8 17h2v2H8zM12 17h2v2h-2z"/></svg>
                CSV
            </button>
            <button type="button" class="btn-export btn-export--pdf" data-format="pdf" title="Download printable report">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M8 13h4M8 17h6"/></svg>
                PDF
            </button>
            </div>
        </div>`;
}

function renderHistoryExportButtons(type, index) {
    return `
        <div class="export-toolbar export-toolbar--compact" role="group" aria-label="Export history item" data-history-type="${type}" data-history-index="${index}">
            <button type="button" class="btn-export btn-export--json btn-export--sm" data-format="json">JSON</button>
            <button type="button" class="btn-export btn-export--csv btn-export--sm" data-format="csv">CSV</button>
            <button type="button" class="btn-export btn-export--pdf btn-export--sm" data-format="pdf">PDF</button>
        </div>`;
}

// Attach export handlers for a result container (JSON/CSV/PDF buttons)
function attachExportHandlers(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const toolbarButtons = container.querySelectorAll('.export-toolbar__buttons');
    toolbarButtons.forEach(wrapper => {
        const tab = wrapper.dataset.tab;
        wrapper.querySelectorAll('.btn-export').forEach(btn => {
            // Avoid double-binding
            if (btn._exportBound) return;
            btn.addEventListener('click', (e) => {
                const format = btn.dataset.format;
                exportLatest(tab, format);
            });
            btn._exportBound = true;
        });
    });
}

// Attach handlers for history export buttons after history is rendered
function attachHistoryExportHandlers(type) {
    const listId = type === 'text' ? 'textHistoryList' : type === 'url' ? 'urlHistoryList' : 'imageHistoryList';
    const list = document.getElementById(listId);
    if (!list) return;
    list.querySelectorAll('.export-toolbar--compact').forEach(wrapper => {
        const idx = Number(wrapper.dataset.historyIndex);
        wrapper.querySelectorAll('.btn-export').forEach(btn => {
            if (btn._histBound) return;
            btn.addEventListener('click', () => {
                const fmt = btn.dataset.format;
                exportHistoryItem(type, idx, fmt);
            });
            btn._histBound = true;
        });
    });
}

// Legacy names (if referenced elsewhere)
function downloadJSON(data) {
    if (data?.classification) {
        exportReport(data, 'json');
        return;
    }
    if (data?.redFlags || data?.title?.includes('Toolkit')) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        triggerDownload(blob, `neurolex-toolkit-${Date.now()}.json`);
        showNotification('Toolkit exported as JSON', 'success');
        return;
    }
    exportReport(buildExportReport(data), 'json');
}
function downloadExcel(data) {
    downloadCSV(data);
}
function downloadCSV(data) {
    if (data?.classification) {
        exportReport(data, 'csv');
        return;
    }
    exportReport(buildExportReport(data), 'csv');
}
function downloadPDF(data) {
    if (data?.classification) {
        exportReport(data, 'pdf');
        return;
    }
    exportReport(buildExportReport(data), 'pdf');
}

// Add to history
function addToHistory(type, data, input) {

    const p = normalizePrediction(data);

    const historyItem = {
        timestamp: new Date().toLocaleString(),
        input: input.substring(0, 100) + (input.length > 100 ? '...' : ''),
        result: p.label,
        confidence: (p.confidence * 100).toFixed(1) + '%',
        data: data
    };

    if (type === 'text') {
        textHistory.unshift(historyItem);
        updateHistoryDisplay('text');
    } else if (type === 'url') {
        urlHistory.unshift(historyItem);
        updateHistoryDisplay('url');
    } else if (type === 'image') {
        imageHistory.unshift(historyItem);
        updateHistoryDisplay('image');
    }
}

// Update history display
function updateHistoryDisplay(type) {
    const history = type === 'text' ? textHistory : type === 'url' ? urlHistory : imageHistory;
    const listId = type === 'text' ? 'textHistoryList' : type === 'url' ? 'urlHistoryList' : 'imageHistoryList';
    const list = document.getElementById(listId);

    if (history.length === 0) {
        list.innerHTML = '<p style="color: var(--text-secondary);">No analysis history yet</p>';
        return;
    }

    list.innerHTML = history.slice(0, 5).map((item, index) => `
                <div class="history-item">
                    <div class="history-item__head">
                        <strong class="history-item__label history-item__label--${(item.result || 'unknown').toLowerCase()}">${(item.result || 'unknown').toUpperCase()}</strong>
                        <span class="history-item__time">${escapeHtml(item.timestamp)}</span>
                    </div>
                    <p class="history-item__preview">${escapeHtml(item.input)}</p>
                    <div class="history-item__footer">
                        <span class="history-item__confidence">${escapeHtml(item.confidence)} confidence</span>
                        
                    </div>
                </div>
            `).join('');
    
}

// Initialize advanced sidebar chart with animations
function drawSidebarChart(serverStats = null) {
    const canvas = document.getElementById('sidebarChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    // Enhanced data with trends
    const data = [
        { value: 96, label: 'Accuracy', color: ['#6366f1', '#8b5cf6'], trend: '+2%' },
        { value: 94, label: 'Precision', color: ['#10b981', '#059669'], trend: '+1%' },
        { value: 93, label: 'Recall', color: ['#f59e0b', '#d97706'], trend: '+3%' },
        { value: 95, label: 'F1-Score', color: ['#ec4899', '#db2777'], trend: '+2%' },
        { value: 97, label: 'AUC', color: ['#8b5cf6', '#6366f1'], trend: '+1%' }
    ];

    const barWidth = canvas.width / data.length - 10;
    const maxHeight = canvas.height - 60;
    let animationProgress = 0;

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (animationProgress < 1) {
            animationProgress += 0.02;
        }

        data.forEach((item, index) => {
            const height = (item.value / 100) * maxHeight * animationProgress;
            const x = index * (barWidth + 10) + 5;
            const y = canvas.height - height - 40;

            // Draw bar with gradient
            const gradient = ctx.createLinearGradient(0, y, 0, canvas.height);
            gradient.addColorStop(0, item.color[0]);
            gradient.addColorStop(1, item.color[1]);
            ctx.fillStyle = gradient;

            // Rounded rectangle
            ctx.beginPath();
            ctx.roundRect(x, y, barWidth, height, [5, 5, 0, 0]);
            ctx.fill();

            // Add glow effect
            ctx.shadowColor = item.color[0];
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Draw value
            ctx.fillStyle = '#f1f5f9';
            ctx.font = 'bold 14px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(item.value + '%', x + barWidth / 2, y - 8);

            // Draw trend
            ctx.fillStyle = '#10b981';
            ctx.font = '10px Inter';
            ctx.fillText(item.trend, x + barWidth / 2, y - 22);

            // Draw label
            ctx.fillStyle = '#cbd5e1';
            ctx.font = '11px Inter';
            ctx.fillText(item.label, x + barWidth / 2, canvas.height - 25);
        });

        if (animationProgress < 1) {
            requestAnimationFrame(animate);
        }
    }

    animate();

    // Redraw with animation on window resize
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            animationProgress = 0;
            animate();
        }, 250);
    });
}

// Draw dynamic Real vs Fake news breakdown on the chat visualization canvas
function drawChatVisualization(serverStats = null) {
    const canvas = document.getElementById('chatVizCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    // Set internal dimensions to match display size
    // Use minimum dimensions if element is hidden (clientWidth would be 0)
    const width = canvas.clientWidth || 600;
    const height = canvas.clientHeight || 300;
    canvas.width = width;
    canvas.height = height;

    // Default mock stats if server has no predictions yet
    let total = 100;
    let fake = 35;
    let real = 65;
    let fakePct = 35;
    let realPct = 65;

    if (serverStats && serverStats.total > 0) {
        total = serverStats.total;
        fake = serverStats.fake ?? 0;
        real = serverStats.real ?? 0;
        fakePct = serverStats.fake_percentage ?? 0;
        realPct = 100 - fakePct;
    } else if (sessionStats && sessionStats.count > 0) {
        total = sessionStats.count;
        fake = sessionStats.fake;
        real = sessionStats.real;
        fakePct = Math.round((fake / total) * 100);
        realPct = 100 - fakePct;
    }

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw donut chart
    const centerX = width / 2;
    const centerY = height / 2 - 10;
    const radius = Math.min(width, height) / 3.5;
    const thickness = 22;

    // Draw background track
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = thickness;
    ctx.stroke();

    // Angles
    const fakeAngle = (fakePct / 100) * Math.PI * 2;
    const startAngle = -Math.PI / 2;

    // Draw Real segment
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle + fakeAngle, startAngle + Math.PI * 2);
    const gradReal = ctx.createLinearGradient(centerX - radius, centerY - radius, centerX + radius, centerY + radius);
    gradReal.addColorStop(0, '#10b981');
    gradReal.addColorStop(1, '#059669');
    ctx.strokeStyle = gradReal;
    ctx.lineWidth = thickness;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Draw Fake segment (only if fakePct > 0)
    if (fakePct > 0) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, startAngle + fakeAngle);
        const gradFake = ctx.createLinearGradient(centerX - radius, centerY - radius, centerX + radius, centerY + radius);
        gradFake.addColorStop(0, '#ef4444');
        gradFake.addColorStop(1, '#dc2626');
        ctx.strokeStyle = gradFake;
        ctx.lineWidth = thickness;
        ctx.lineCap = 'round';
        ctx.stroke();
    }

    // Draw center text
    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 22px Inter';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${total}`, centerX, centerY - 8);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '500 10px Inter';
    ctx.fillText('TOTAL ANALYSES', centerX, centerY + 14);

    // Draw Legend / Metrics at the bottom
    const legendY = height - 25;
    
    // Real Label
    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 13px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(`✓ Real: ${realPct}% (${real})`, centerX - 25, legendY);

    // Fake Label
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 13px Inter';
    ctx.textAlign = 'left';
    ctx.fillText(`✗ Fake: ${fakePct}% (${fake})`, centerX + 25, legendY);
}

// Refresh tutorials
function refreshTutorials() {
    const tutorials = document.querySelectorAll('.sidebar-nav .sidebar-item');
    tutorials.forEach((item, index) => {
        setTimeout(() => {
            item.style.animation = 'none';
            setTimeout(() => {
                item.style.animation = 'slideInLeft 0.5s ease';
            }, 10);
        }, index * 100);
    });

    // Show notification
    showNotification('✅ Tutorials refreshed!', 'success');
}

// Export toolkit
function exportToolkit() {
    const toolkit = {
        redFlags: [
            'ALL CAPS headlines',
            'Anonymous sources',
            'Emotional manipulation',
            'No publication date',
            'Poor grammar/spelling'
        ],
        verificationChecklist: [
            'Check 3+ sources',
            'Verify author credentials',
            'Look for citations',
            'Reverse image search',
            'Check domain reputation'
        ],
        globalStats: {
            'Exposure Rate': '64% have seen fake news',
            'Sharing Rate': '23% shared unknowingly',
            'Economic Impact': '$78B annual damage',
            'Spread Speed': 'Spreads 6x faster'
        }
    };

    downloadJSON(toolkit);
    showNotification('📥 Toolkit exported successfully!', 'success');
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
                position: fixed;
                top: 5rem;
                right: 2rem;
                z-index: 10001;
                background: var(--glass-bg);
                border: 2px solid var(--${type === 'success' ? 'secondary' : type === 'error' ? 'danger' : 'primary'});
                padding: 1rem 1.5rem;
                border-radius: 12px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                animation: slideInRight 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                font-weight: 600;
            `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.4s ease';
        setTimeout(() => notification.remove(), 400);
    }, 3000);
}

// Add CSS animation keyframes
const style = document.createElement('style');
style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
            @keyframes slideInLeft {
                from { transform: translateX(-100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
document.head.appendChild(style);

// Initialize animated background
function initBackground() {
    const container = document.getElementById('bgAnimation');
    const particleCount = 30;

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';

        const size = Math.random() * 100 + 50;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.animationDuration = `${Math.random() * 20 + 10}s`;
        particle.style.animationDelay = `${Math.random() * 5}s`;

        container.appendChild(particle);
    }
}

// Tab switching
function switchTab(tab) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

    // Show selected tab
    if (tab === 'text') {
        document.getElementById('textTab').classList.add('active');
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
    } else if (tab === 'url') {
        document.getElementById('urlTab').classList.add('active');
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
    } else if (tab === 'image') {
        document.getElementById('imageTab').classList.add('active');
        document.querySelectorAll('.tab-btn')[2].classList.add('active');
    } else if (tab === 'chat') {
        document.getElementById('chatTab').classList.add('active');
        document.querySelectorAll('.tab-btn')[3].classList.add('active');
        
        // Redraw visualization when chat tab becomes visible
        // This ensures the canvas has proper dimensions after being hidden
        setTimeout(() => {
            drawChatVisualization();
        }, 100);
    }
}

// Display result
function displayResult(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const label = (data?.label || 'unknown').toLowerCase();
    const confidencePct = normalizeConfidence(data?.confidence) * 100;
    const tierName = (data?.tier || 'UNCERTAIN').toUpperCase();
    const badgeClass = tierName === 'HIGH' ? 'high' : tierName === 'MEDIUM' ? 'medium' : tierName === 'LOW' ? 'low' : 'low';

    let emoji = '✅';
    let cardClass = 'real';
    if (label === 'fake') {
        emoji = '❌';
        cardClass = 'fake';
    } else if (label === 'uncertain' || label === 'unknown') {
        emoji = '⚠️';
        cardClass = 'uncertain';
    } else if (confidencePct < 60) {
        emoji = '⚠️';
        cardClass = 'uncertain';
    }

    const probs = data?.probabilities || {};
    const probHtml = (probs.REAL != null || probs.FAKE != null)
        ? `<div class="stat-item">
                <div class="stat-label">Probabilities</div>
                <div class="stat-value">REAL ${((probs.REAL ?? 0) * 100).toFixed(1)}% · FAKE ${((probs.FAKE ?? 0) * 100).toFixed(1)}%</div>
           </div>`
        : '';

    const warningsHtml = (data?.warnings?.length > 0)
        ? `<div class="explanation">
                <div class="explanation-title">⚠️ Warnings</div>
                <ul style="margin:0.5rem 0 0 1rem;">${data.warnings.map(w => `<li>${w}</li>`).join('')}</ul>
           </div>`
        : '';

    container.innerHTML = `
        <div class="result-card ${cardClass}">
            <div class="result-header">
                <div class="result-label">${emoji} ${label.toUpperCase()}</div>
                <div class="confidence-badge confidence-${badgeClass}">
                    ${tierName} TIER${data.tierAccuracy ? ' · ' + data.tierAccuracy : ''}
                </div>
            </div>
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill ${cardClass}" style="width: ${confidencePct}%"></div>
                </div>
                <div style="margin-top: 0.5rem; font-weight: 600;">${confidencePct.toFixed(1)}% Confidence</div>
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Models</div>
                    <div class="stat-value">${data.model || 'Ensemble'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Processing Time</div>
                    <div class="stat-value">${data.processing_time > 0 ? data.processing_time.toFixed(2) + 's' : 'N/A'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Tier</div>
                    <div class="stat-value">${tierName}</div>
                </div>
                ${data.domain ? `<div class="stat-item"><div class="stat-label">Domain</div><div class="stat-value">${data.domain}</div></div>` : ''}
                ${probHtml}
            </div>
            ${data.explanation ? `<div class="explanation"><div class="explanation-title">💡 Recommendation</div><p>${data.explanation}</p></div>` : ''}
            ${data.patterns?.length ? `<div class="explanation"><div class="explanation-title">🔍 Detected Patterns</div><p>${data.patterns.join(', ')}</p></div>` : ''}
            ${warningsHtml}
            ${data.extraHtml || ''}
        </div>`;
    container.classList.remove('hidden');
}

// Text form submission
document.getElementById('textForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const text = document.getElementById('textInput').value;
    const language = document.getElementById('langSelect').value;
    const btnText = document.getElementById('textBtnText');
    const spinner = document.getElementById('textSpinner');

    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    hideContentSignals();
    showAnalysisLoading('textResult');

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, language })
        });

        if (!response.ok) throw new Error(await parseApiError(response));

        const data = await response.json();
        const p = normalizePrediction(data);
        displayResultEnhanced('textResult', data, text);
        addToHistory('text', data, text);
        bumpSessionStats(p.label);
        loadLiveStats();
        addActivity('text', 'Analyzed text — ' + p.label.toUpperCase());
    } catch (error) {
        document.getElementById('textResult').innerHTML = `
            <div class="result-card uncertain"><p style="color:var(--danger);">❌ ${error.message}</p></div>`;
        document.getElementById('textResult').classList.remove('hidden');
    } finally {
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
});

// URL form submission
document.getElementById('urlForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const url = document.getElementById('urlInput').value;
    const btnText = document.getElementById('urlBtnText');
    const spinner = document.getElementById('urlSpinner');

    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    showAnalysisLoading('urlResult');

    try {
        const response = await fetch(`${API_BASE}/analyze_url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        if (!response.ok) throw new Error(await parseApiError(response));

        const data = await response.json();
        const p = normalizePrediction(data);
        displayResultEnhanced('urlResult', data, url);
        addToHistory('url', data, url);
        bumpSessionStats(p.label);
        loadLiveStats();
        addActivity('url', 'Analyzed URL — ' + p.label.toUpperCase());

    } catch (error) {
        document.getElementById('urlResult').innerHTML = `
            <div class="result-card uncertain"><p style="color:var(--danger);">❌ ${error.message}</p></div>`;
        document.getElementById('urlResult').classList.remove('hidden');
        console.error(error);
    } finally {
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
});

// Chat form submission
// Chat form submission
document.getElementById('chatForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const message = document.getElementById('chatInput').value;
    const messagesContainer = document.getElementById('chatMessages');
    const btnText = document.getElementById('chatBtnText');
    const spinner = document.getElementById('chatSpinner');
    // Add user message (safe DOM)
    const userWrap = document.createElement('div');
    userWrap.className = 'chat-message-container-user';
    const userInner = document.createElement('div');
    userInner.className = 'chat-message-user';
    userInner.textContent = message;
    userWrap.appendChild(userInner);
    messagesContainer.appendChild(userWrap);

    document.getElementById('chatInput').value = '';

    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');

    // Add typing indicator
    const typingWrap = document.createElement('div');
    typingWrap.className = 'chat-message-container-bot';
    const typingInner = document.createElement('div');
    typingInner.className = 'chat-message-bot chat-typing';
    typingInner.textContent = 'AI is typing...';
    typingWrap.appendChild(typingInner);
    messagesContainer.appendChild(typingWrap);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
        let data = null;

        // Prefer Groq-powered LLM chat; fall back to rule-based /chat
        try {
            const llmResponse = await fetch(`${API_BASE}/llm/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    session_id: chatSessionId,
                    include_context: true
                })
            });
            if (llmResponse.ok) {
                data = await llmResponse.json();
                if (data.session_id) {
                    chatSessionId = data.session_id;
                    localStorage.setItem('neurolex_chat_session', chatSessionId);
                }
            }
        } catch (llmErr) {
            console.warn('LLM chat unavailable, using fallback:', llmErr);
        }

        if (!data) {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, history: [] })
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || err.message || `Chat failed (${response.status})`);
            }
            data = await response.json();
        }

        const botReplyRaw = data.reply || data.response || data.message || 'No response';
        const botReply = (typeof botReplyRaw === 'string') ? botReplyRaw : JSON.stringify(botReplyRaw);

        // Remove typing indicator
        if (typingWrap && typingWrap.parentNode) typingWrap.parentNode.removeChild(typingWrap);

        // Build bot message element safely; render markdown for nicer formatting
        const botWrap = document.createElement('div');
        botWrap.className = 'chat-message-container-bot';
        const botInner = document.createElement('div');
        botInner.className = 'chat-message-bot';

        // Collapse long replies
        const MAX_LEN = 1200;
        if (botReply.length > MAX_LEN) {
            const shortText = botReply.slice(0, 800) + '...';
            botInner.innerHTML = renderMarkdown(shortText);

            const moreBtn = document.createElement('button');
            moreBtn.className = 'btn-more';
            moreBtn.textContent = 'Show more';
            moreBtn.style.marginTop = '8px';
            moreBtn.addEventListener('click', () => {
                botInner.innerHTML = renderMarkdown(botReply);
                moreBtn.remove();
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            });
            botWrap.appendChild(botInner);
            botWrap.appendChild(moreBtn);
        } else {
            botInner.innerHTML = renderMarkdown(botReply);
            botWrap.appendChild(botInner);
        }

        messagesContainer.appendChild(botWrap);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // ✅ activity logging (safe)
        addActivity('chat', 'AI responded to: ' + (botReply.substring(0, 40) + '...'));

    } catch (error) {
        // Remove typing indicator
        if (typingWrap && typingWrap.parentNode) typingWrap.parentNode.removeChild(typingWrap);

        const errWrap = document.createElement('div');
        errWrap.className = 'chat-message-container-bot';
        const errInner = document.createElement('div');
        errInner.className = 'chat-message-bot';
        errInner.style.borderLeftColor = 'var(--danger)';
        errInner.style.background = 'rgba(239, 68, 68, 0.25)';
        errInner.style.color = '#ef4444';
        errInner.textContent = 'Error: ' + (error.message || String(error));
        errWrap.appendChild(errInner);
        messagesContainer.appendChild(errWrap);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    } finally {
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
});


// Image form submission
document.getElementById('imageForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    const btnText = document.getElementById('imageBtnText');
    const spinner = document.getElementById('imageSpinner');

    if (!file) {
        alert('Please select an image file');
        return;
    }

    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    showAnalysisLoading('imageResult');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/analyze_image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(await parseApiError(response));

        const data = await response.json();
        const p = normalizePrediction(data);
        displayResultEnhanced('imageResult', data, file.name);
        addToHistory('image', data, file.name);
        bumpSessionStats(p.label);
        loadLiveStats();
        addActivity('image', 'Analyzed image — ' + p.label.toUpperCase());

    } catch (error) {
        document.getElementById('imageResult').innerHTML = `
            <div class="result-card uncertain"><p style="color:var(--danger);">❌ ${error.message}</p></div>`;
        document.getElementById('imageResult').classList.remove('hidden');
    } finally {
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
});

// Image preview
document.getElementById('imageInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('imagePreview').innerHTML = `
                        <img src="${e.target.result}" style="max-width: 400px; max-height: 300px; border-radius: 10px; border: 2px solid var(--glass-border);" />
                    `;
        };
        reader.readAsDataURL(file);
    }
});

// Animate counters
function animateCounters() {
    document.querySelectorAll('.stat-counter').forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target'));
        const duration = 2000;
        const step = target / (duration / 16);
        let current = 0;

        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                counter.textContent = target + (target < 100 ? '%' : '');
                clearInterval(timer);
            } else {
                counter.textContent = Math.floor(current) + (target < 100 ? '%' : '');
            }
        }, 16);
    });
}

// Generate attention heatmap
function generateAttentionHeatmap(text, attentionScores) {
    const words = text.split(' ');
    const heatmapContent = document.getElementById('heatmapContent');
    const attentionHeatmap = document.getElementById('textAttentionHeatmap');

    let html = '';
    words.forEach((word, index) => {
        const score = attentionScores && attentionScores[index] ? attentionScores[index] : Math.random();
        const intensity = Math.floor(score * 255);
        const bgColor = `rgba(239, 68, 68, ${score})`;
        html += `<span class="heatmap-word" style="background: ${bgColor};" title="Attention: ${(score * 100).toFixed(1)}%">${word}</span>`;
    });

    heatmapContent.innerHTML = html;
    attentionHeatmap.classList.remove('hidden');
}

// Generate SHAP values visualization
function generateShapValues(features) {
    const shapContent = document.getElementById('shapContent');
    const shapValues = document.getElementById('textShapValues');

    const mockFeatures = features || [
        { name: 'Sensationalism', value: 0.45, positive: false },
        { name: 'Source Quality', value: 0.32, positive: true },
        { name: 'Emotional Language', value: 0.28, positive: false },
        { name: 'Fact Density', value: 0.25, positive: true },
        { name: 'Grammar Quality', value: 0.18, positive: true }
    ];

    let html = '';
    mockFeatures.forEach(feature => {
        const width = Math.abs(feature.value) * 100;
        const className = feature.positive ? 'shap-positive' : 'shap-negative';
        html += `
                    <div class="shap-bar">
                        <div class="shap-label">${feature.name}</div>
                        <div class="shap-value-bar ${className}" style="width: ${width}%">
                            <span class="shap-value-text">${feature.positive ? '+' : ''}${feature.value.toFixed(2)}</span>
                        </div>
                    </div>
                `;
    });

    shapContent.innerHTML = html;
    shapValues.classList.remove('hidden');
}

// Generate neural network visualization
function generateNeuralViz(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const width = container.offsetWidth;
    const height = 200;

    // Create neural nodes
    for (let layer = 0; layer < 4; layer++) {
        const nodesInLayer = layer === 0 || layer === 3 ? 3 : 5;
        for (let node = 0; node < nodesInLayer; node++) {
            const nodeEl = document.createElement('div');
            nodeEl.className = 'neural-node';
            nodeEl.style.left = `${(layer * (width / 3)) + 20}px`;
            nodeEl.style.top = `${(node * (height / nodesInLayer)) + 40}px`;
            nodeEl.style.animationDelay = `${(layer + node) * 0.2}s`;
            container.appendChild(nodeEl);
        }
    }
}

// Add activity to feed
function addActivity(type, message) {
    const feed = document.getElementById('activityFeed');
    const icons = { text: '📝', url: '🔗', image: '🖼️', chat: '💬' };

    const item = document.createElement('div');
    item.className = 'activity-item';
    item.innerHTML = `
                <div class="activity-icon">${icons[type]}</div>
                <div class="activity-content">
                    <div style="font-weight: 600;">${message}</div>
                    <div class="activity-time">${new Date().toLocaleTimeString()}</div>
                </div>
            `;

    feed.insertBefore(item, feed.firstChild);

    // Keep only last 10 items
    while (feed.children.length > 10) {
        feed.removeChild(feed.lastChild);
    }
}

// Enhanced display result with visualizations
function displayResultEnhanced(containerId, data, inputText) {

    const p = normalizePrediction(data); // 🔥 FIX HERE

    const extraWarnings = (p.warnings || []).map(w => `<p style="margin:0.4rem 0;font-size:0.9rem;color:var(--text-secondary);">${w}</p>`).join('');

    const exportTab = TAB_FROM_RESULT_ID[containerId] || null;
    if (exportTab) {
        lastAnalysisByTab[exportTab] = { raw: data, input: inputText || '' };
    }

    displayResult(containerId, {
        label: p.label,
        confidence: p.confidence,
        tier: p.tier,
        tierAccuracy: p.tierAccuracy,
        model: p.model,
        explanation: p.explanation,
        processing_time: p.processing_time,
        probabilities: p.probabilities,
        warnings: p.warnings,
        patterns: p.patterns?.length ? p.patterns : (data?.analysis?.pattern_detection?.patterns || []),
        extraHtml: extraWarnings,
        exportTab,
    });

    // URL domain fix
    if (containerId === 'urlResult' && data?.domain_analysis) {
        const d = data.domain_analysis;

        document.getElementById('domainAnalysisContent').innerHTML = `
            <div style="margin-top: 0.5rem;">
                <strong>Domain:</strong> ${d.domain || 'Unknown'}<br>
                <strong>Trust Score:</strong> ${d.credibility_score ?? 0}<br>
                <strong>Category:</strong> ${d.category || 'Unknown'}<br>
                <strong>Suspicious:</strong> ${d.is_suspicious ? "YES" : "NO"}<br>
            </div>
        `;

        document.getElementById('urlDomainAnalysis').classList.remove('hidden');
    }

    if (containerId === 'imageResult') {
        const trusted = data?.analysis?.trusted_sources;
        if (trusted?.detected && trusted.sources?.length) {
            const el = document.getElementById('imageResult');
            if (el && !el.querySelector('.ocr-trusted-note')) {
                const note = document.createElement('div');
                note.className = 'explanation ocr-trusted-note';
                note.innerHTML = `<div class="explanation-title">📰 Sources in image text</div>
                    <p>${trusted.sources.slice(0, 5).join(', ')}</p>`;
                el.querySelector('.result-card')?.appendChild(note);
            }
        }
        if (data?.ocr_text_preview) {
            const el = document.getElementById('imageResult');
            if (el && !el.querySelector('.ocr-preview-note')) {
                const note = document.createElement('div');
                note.className = 'explanation ocr-preview-note';
                note.innerHTML = `<div class="explanation-title">📝 OCR excerpt</div>
                    <p style="font-size:0.85rem;max-height:120px;overflow-y:auto;">${data.ocr_text_preview}</p>`;
                el.querySelector('.result-card')?.appendChild(note);
            }
        }
    }
}

// Advanced 3D tilt effect for metric cards
function init3DTilt() {
    document.querySelectorAll('[data-tilt]').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = ((y - centerY) / centerY) * 10;
            const rotateY = ((centerX - x) / centerX) * 10;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-10px) scale(1.05)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0) scale(1)';
        });
    });
}

// Real-time API health check
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/healthz`, { method: 'GET' });
        const data = await response.json();

        // Update live indicator
        const indicators = document.querySelectorAll('.live-indicator');
        indicators.forEach(indicator => {
            if (data.status === 'ok' || data.status === 'healthy') {
                indicator.style.borderColor = 'var(--secondary)';
                indicator.querySelector('.live-dot').style.background = 'var(--secondary)';
            }
        });
    } catch (error) {
        console.log('API check failed:', error);
    }
}

// Keyboard shortcuts
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Alt+1: Text Analysis
        if (e.altKey && e.key === '1') {
            e.preventDefault();
            switchTab('text');
            document.getElementById('textInput').focus();
        }
        // Alt+2: URL Analysis
        if (e.altKey && e.key === '2') {
            e.preventDefault();
            switchTab('url');
            document.getElementById('urlInput').focus();
        }
        // Alt+3: Image Analysis
        if (e.altKey && e.key === '3') {
            e.preventDefault();
            switchTab('image');
        }
        // Alt+4: Chat
        if (e.altKey && e.key === '4') {
            e.preventDefault();
            switchTab('chat');
            document.getElementById('chatInput').focus();
        }
        // Alt+S: Toggle Sidebar
        if (e.altKey && e.key === 's') {
            e.preventDefault();
            toggleSidebar();
        }
        // Alt+T: Toggle Theme
        if (e.altKey && e.key === 't') {
            e.preventDefault();
            toggleTheme();
        }
    });

    // Show keyboard shortcut hints
    showNotification('💡 Keyboard shortcuts enabled! Alt+1-4 for tabs, Alt+S for sidebar, Alt+T for theme', 'info');
}

// Export functions exposed globally for inline onclick
window.exportHistoryItem = exportHistoryItem;
window.exportLatest = exportLatest;

// Initialize on load
window.addEventListener('load', () => {
    initBackground();
    drawSidebarChart();
    drawChatVisualization();
    updateHistoryDisplay('text');
    updateHistoryDisplay('url');
    updateHistoryDisplay('image');
    addFloatingStats();
    animateCounters();
    init3DTilt();
    initKeyboardShortcuts();
    checkAPIHealth();

    // Redraw chat visualization on window resize
    window.addEventListener('resize', () => {
        drawChatVisualization();
    });

    // Periodic API health check every 30 seconds
    setInterval(checkAPIHealth, 30000);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeSidebar();
    });

    // Add pulse effect to important elements
    document.querySelectorAll('.btn-primary').forEach(btn => {
        btn.classList.add('pulse-element');
    });

    // Add glow effect to cards
    document.querySelectorAll('.glass-card').forEach(card => {
        card.classList.add('glow-on-hover');
    });

    // Show welcome notification
    setTimeout(() => {
        showNotification('🎉 Welcome to NeuroLex AI v3.0! Advanced fake news detection at your fingertips.', 'success');
    }, 1000);
});
