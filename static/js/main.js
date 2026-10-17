document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const submitBtn = document.getElementById('submit-btn');
  const spinner = document.getElementById('spinner');
  const btnText = submitBtn ? submitBtn.querySelector('.btn-text') : null;
  const resetBtn = document.getElementById('reset-btn');

  const emptyState = document.getElementById('empty-state');
  const predictionContent = document.getElementById('prediction-content');
  const errorAlert = document.getElementById('error-alert');
  const errorText = document.getElementById('error-text');

  const resultBadge = document.getElementById('result-badge');
  const predictedClass = document.getElementById('predicted-class');
  const cultivarDescription = document.getElementById('cultivar-description');
  const confidenceVal = document.getElementById('confidence-val');
  const meterFill = document.getElementById('meter-fill');
  const probList = document.getElementById('prob-list');

  const presetButtons = document.querySelectorAll('.preset-pill, .preset-btn');

  // Handle Preset Click
  presetButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const presetKey = btn.getAttribute('data-preset');
      const presets = window.PRESETS_DATA || {};
      const preset = presets[presetKey];

      if (!preset || !preset.features) {
        console.warn('Preset not found for key:', presetKey);
        return;
      }

      // Fill in all input fields
      Object.entries(preset.features).forEach(([field, val]) => {
        const input = document.getElementById(field);
        if (input) {
          input.value = val;
        }
      });

      // Highlight active pill button
      presetButtons.forEach((b) => {
        b.style.borderColor = 'var(--border, #e2e8f0)';
        b.style.background = 'var(--bg-page, #f8fafc)';
        b.style.color = 'var(--text-secondary, #475569)';
      });
      btn.style.borderColor = 'var(--primary-dark, #059669)';
      btn.style.background = 'var(--primary-subtle, #f0fdf4)';
      btn.style.color = 'var(--primary-dark, #059669)';

      // Trigger classification
      executePrediction();
    });
  });

  // Handle Form Reset
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      if (form) form.reset();
      if (emptyState) emptyState.classList.remove('hidden');
      if (predictionContent) predictionContent.classList.add('hidden');
      if (errorAlert) errorAlert.classList.add('hidden');
      presetButtons.forEach((b) => {
        b.style.borderColor = 'var(--border, #e2e8f0)';
        b.style.background = 'var(--bg-page, #f8fafc)';
        b.style.color = 'var(--text-secondary, #475569)';
      });
    });
  }

  // Handle Manual Form Submission
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      executePrediction();
    });
  }

  // Core Prediction Function
  async function executePrediction() {
    if (!form) return;

    // Collect 16 feature values
    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
      payload[key] = parseFloat(value);
    }

    setLoading(true);
    if (errorAlert) errorAlert.classList.add('hidden');

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok || data.status !== 'success') {
        throw new Error(data.message || 'Server returned an error.');
      }

      renderResults(data);

    } catch (err) {
      console.error('Prediction request failed:', err);
      showError(err.message || 'Failed to connect to the classification service.');
    } finally {
      setLoading(false);
    }
  }

  function renderResults(data) {
    if (emptyState) emptyState.classList.add('hidden');
    if (predictionContent) predictionContent.classList.remove('hidden');

    // Winner details
    if (predictedClass) {
      predictedClass.textContent = data.prediction;
      predictedClass.style.color = data.badge_color || '#059669';
    }
    if (resultBadge) {
      resultBadge.textContent = `Cultivar: ${data.display_name || data.prediction}`;
      resultBadge.style.color = data.badge_color || '#059669';
    }
    if (cultivarDescription) {
      cultivarDescription.textContent = data.description || '';
    }

    // Confidence meter
    const conf = parseFloat(data.confidence) || 0;
    if (confidenceVal) {
      confidenceVal.textContent = `${conf.toFixed(1)}%`;
    }
    if (meterFill) {
      meterFill.style.width = `${Math.min(conf, 100)}%`;
    }

    // Multiclass Probabilities
    if (probList) {
      probList.innerHTML = '';
      const probs = data.probabilities || {};

      Object.entries(probs).forEach(([cls, prob]) => {
        const isWinner = cls === data.prediction;
        const item = document.createElement('div');
        item.className = `prob-item ${isWinner ? 'winner' : ''}`;

        item.innerHTML = `
          <div class="prob-label-row">
            <span class="prob-name">${cls}</span>
            <span class="prob-pct">${prob.toFixed(1)}%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width: ${Math.min(prob, 100)}%"></div>
          </div>
        `;
        probList.appendChild(item);
      });
    }

    // Smooth scroll into view on small screens
    if (window.innerWidth < 860 && predictionContent) {
      predictionContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function showError(msg) {
    if (errorText) errorText.textContent = msg;
    if (errorAlert) errorAlert.classList.remove('hidden');
    if (predictionContent) predictionContent.classList.add('hidden');
    if (emptyState) emptyState.classList.remove('hidden');
  }

  function setLoading(isLoading) {
    if (!submitBtn) return;
    if (isLoading) {
      submitBtn.disabled = true;
      if (spinner) spinner.classList.remove('hidden');
      if (btnText) btnText.textContent = 'Classifying...';
    } else {
      submitBtn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
      if (btnText) btnText.textContent = 'Classify Bean';
    }
  }
});
