const API_BASE = '';  // same origin; change to 'http://localhost:8000' for dev

const textarea     = document.getElementById('article-input');
const classifyBtn  = document.getElementById('classify-btn');
const clearBtn     = document.getElementById('clear-btn');
const spinner      = document.getElementById('spinner');
const errorBox     = document.getElementById('error-box');
const resultsSection = document.getElementById('results-section');
const categoryBadge  = document.getElementById('category-badge');
const confidenceLabel = document.getElementById('confidence-label');
const scoresContainer = document.getElementById('scores-container');

function showSpinner(show) {
  spinner.classList.toggle('hidden', !show);
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
}

function hideError() {
  errorBox.classList.add('hidden');
  errorBox.textContent = '';
}

function showResults(data) {
  const { category, confidence, all_scores } = data;

  // Badge
  categoryBadge.textContent = category;
  categoryBadge.className = `category-badge badge-${category.toLowerCase()}`;

  // Confidence
  confidenceLabel.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;

  // Score bars
  scoresContainer.innerHTML = '';
  const sorted = Object.entries(all_scores).sort((a, b) => b[1] - a[1]);
  const topCat = sorted[0][0];

  sorted.forEach(([cat, score]) => {
    const pct = (score * 100).toFixed(1);
    const isTop = cat === topCat;

    const row = document.createElement('div');
    row.className = 'score-row';

    row.innerHTML = `
      <span class="score-label">${cat}</span>
      <div class="score-bar-track">
        <div class="score-bar-fill ${isTop ? 'top' : ''}" style="width:${pct}%"></div>
      </div>
      <span class="score-value">${pct}%</span>
    `;
    scoresContainer.appendChild(row);
  });

  resultsSection.classList.remove('hidden');
}

async function classify() {
  const text = textarea.value.trim();
  if (!text) {
    showError('Please enter some article text before classifying.');
    return;
  }

  hideError();
  resultsSection.classList.add('hidden');
  showSpinner(true);
  classifyBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    showResults(data);
  } catch (err) {
    showError(`Classification failed: ${err.message}`);
  } finally {
    showSpinner(false);
    classifyBtn.disabled = false;
  }
}

classifyBtn.addEventListener('click', classify);

clearBtn.addEventListener('click', () => {
  textarea.value = '';
  hideError();
  resultsSection.classList.add('hidden');
  textarea.focus();
});

// Allow Ctrl+Enter to submit
textarea.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    classify();
  }
});
