document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const submitBtn = document.getElementById('submit-btn');
  const spinner = document.getElementById('spinner');
  const btnText = submitBtn.querySelector('.btn-text');
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

  const presetButtons = document.querySelectorAll('.preset-btn');

  // Handle Preset Click
  presetButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const presetKey = btn.getAttribute('data-preset');
      const presets = window.PRESETS_DATA || {};
      const preset = presets[presetKey];

      if (!preset || !preset.features) return;

      // Fill in all input fields
      Object.entries(preset.features).forEach(([field, val]) => {
        const input = document.getElementById(field);
        if (input) {
          input.value = val;
        }
      });

      // Highlight active button briefly
      presetButtons.forEach(b => b.style.borderColor = '#e2e8f0');
      btn.style.borderColor = '#059669';

      // Auto-submit after loading preset
      form.dispatchEvent(new Event('submit'));
    });
  });

  // Handle Form Reset
  resetBtn.addEventListener('click', () => {
    form.reset();
    emptyState.classList.remove('hidden');
    predictionContent.classList.add('hidden');
    errorAlert.classList.add('hidden');
    presetButtons.forEach(b => b.style.borderColor = '#e2e8f0');
  });

  // Handle Form Submission via Fetch
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect 16 feature values
    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
      payload[key] = parseFloat(value);
    }

    // Set Loading State
    setLoading(true);
    errorAlert.classList.add('hidden');

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

      // Render Results
      renderResults(data);

    } catch (err) {
      console.error('Prediction request failed:', err);
      showError(err.message || 'Failed to connect to the classification service.');
    } finally {
      setLoading(false);
    }
  });

  function renderResults(data) {
    emptyState.classList.add('hidden');
    predictionContent.classList.remove('hidden');

    // Winner details
    predictedClass.textContent = data.prediction;
    predictedClass.style.color = data.badge_color || '#059669';
    resultBadge.textContent = `Cultivar: ${data.display_name}`;
    resultBadge.style.background = '#ecfdf5';
    resultBadge.style.color = data.badge_color || '#059669';
    cultivarDescription.textContent = data.description || '';

    // Confidence
    const conf = parseFloat(data.confidence) || 0;
    confidenceVal.textContent = `${conf.toFixed(1)}%`;
    meterFill.style.width = `${Math.min(conf, 100)}%`;

    // Multiclass Probabilities
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

    // Smooth scroll into results view on mobile
    if (window.innerWidth < 960) {
      predictionContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function showError(msg) {
    errorText.textContent = msg;
    errorAlert.classList.remove('hidden');
    predictionContent.classList.add('hidden');
    emptyState.classList.remove('hidden');
  }

  function setLoading(isLoading) {
    if (isLoading) {
      submitBtn.disabled = true;
      spinner.classList.remove('hidden');
      btnText.textContent = 'Analyzing Morphology...';
    } else {
      submitBtn.disabled = false;
      spinner.classList.add('hidden');
      btnText.textContent = 'Predict Bean Variety';
    }
  }
});
