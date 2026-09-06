(() => {
  const sceneInput = document.getElementById('sceneInput');
  const adInput = document.getElementById('adInput');
  const videoInput = document.getElementById('videoInput');
  const scenePreview = document.getElementById('scenePreview');
  const adPreview = document.getElementById('adPreview');
  const videoPreview = document.getElementById('videoPreview');

  const confSlider = document.getElementById('confSlider');
  const confVal = document.getElementById('confVal');
  const fitShape = document.getElementById('fitShape');
  const blendEdges = document.getElementById('blendEdges');

  const detectBtn = document.getElementById('detectBtn');
  const videoBtn = document.getElementById('videoBtn');
  const detectError = document.getElementById('detectError');
  const videoError = document.getElementById('videoError');

  const detectionsPanel = document.getElementById('detectionsPanel');
  const detectPreview = document.getElementById('detectPreview');
  const boxSelect = document.getElementById('boxSelect');
  const compositeError = document.getElementById('compositeError');

  const resultPanel = document.getElementById('resultPanel');
  const resultPreview = document.getElementById('resultPreview');
  const resultFinal = document.getElementById('resultFinal');
  const fallbackNote = document.getElementById('fallbackNote');
  const downloadBtn = document.getElementById('downloadBtn');

  const videoResultPanel = document.getElementById('videoResultPanel');
  const videoResult = document.getElementById('videoResult');
  const videoStats = document.getElementById('videoStats');
  const videoDownloadBtn = document.getElementById('videoDownloadBtn');

  const jobNo = document.getElementById('jobNo');
  const jobStatus = document.getElementById('jobStatus');
  const loadingIndicator = document.getElementById('loadingIndicator');

  let boxes = [];
  let compositeToken = 0;

  jobNo.textContent = String(Date.now()).slice(-6);

  function setLoading(isLoading) {
    loadingIndicator.hidden = !isLoading;
  }

  function setStatus(text) {
    jobStatus.textContent = text;
  }

  function showError(el, message) {
    el.textContent = message;
    el.hidden = false;
  }

  function hideError(el) {
    el.hidden = true;
  }

  function updateDetectAvailability() {
    detectBtn.disabled = !(sceneInput.files[0] && adInput.files[0]);
  }

  function updateVideoAvailability() {
    videoBtn.disabled = !(videoInput.files[0] && adInput.files[0]);
  }

  function resetDownstream() {
    boxes = [];
    boxSelect.innerHTML = '';
    detectionsPanel.hidden = true;
    resultPanel.hidden = true;
    fallbackNote.hidden = true;
    hideError(detectError);
    hideError(compositeError);
  }

  function wireUpload(input, previewImg, dropLabel) {
    input.addEventListener('change', () => {
      const file = input.files[0];
      if (!file) return;

      const url = URL.createObjectURL(file);
      previewImg.src = url;
      previewImg.hidden = false;
      dropLabel.hidden = true;

      updateDetectAvailability();
      updateVideoAvailability();
      resetDownstream();
      setStatus('awaiting detection');
    });
  }

  wireUpload(sceneInput, scenePreview, document.querySelector('#sceneFrame .frame__drop'));
  wireUpload(adInput, adPreview, document.querySelector('#adFrame .frame__drop'));

  videoInput.addEventListener('change', () => {
    const file = videoInput.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    videoPreview.hidden = false;
    document.querySelector('#videoFrame .frame__drop').hidden = true;

    updateDetectAvailability();
    updateVideoAvailability();
    videoResultPanel.hidden = true;
    hideError(videoError);
    setStatus('video ready');
  });

  confSlider.addEventListener('input', () => {
    confVal.textContent = Number(confSlider.value).toFixed(2);
  });

  detectBtn.addEventListener('click', async () => {
    hideError(detectError);
    const sceneFile = sceneInput.files[0];
    if (!sceneFile) return;

    setLoading(true);
    setStatus('detecting...');
    detectBtn.disabled = true;
    resultPanel.hidden = true;
    fallbackNote.hidden = true;

    try {
      const form = new FormData();
      form.append('scene', sceneFile);
      form.append('conf', confSlider.value);

      const res = await fetch('/api/detect', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Detection failed (${res.status})`);
      }

      const data = await res.json();
      boxes = data.boxes || [];

      if (boxes.length === 0) {
        showError(detectError, 'No billboard detected above this confidence threshold. Try lowering it, or use a clearer scene photo.');
        detectionsPanel.hidden = true;
        resultPanel.hidden = true;
        setStatus('no detections');
        return;
      }

      detectPreview.src = data.preview;
      boxSelect.innerHTML = '';

      boxes.forEach((b, i) => {
        const opt = document.createElement('option');
        opt.value = String(i);
        const shapeLabel = b.used_fallback ? 'rectangle fallback' : '4-corner shape';
        opt.textContent = `detection ${i + 1} - confidence ${b.confidence.toFixed(2)} - ${shapeLabel} - box (${b.x1},${b.y1})-(${b.x2},${b.y2})`;
        boxSelect.appendChild(opt);
      });

      boxSelect.value = '0';
      detectionsPanel.hidden = false;
      setStatus(`${boxes.length} panel(s) found`);
      detectionsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      runComposite({ scroll: true });
    } catch (e) {
      showError(detectError, e.message || 'Something went wrong running detection.');
      setStatus('detection error');
    } finally {
      setLoading(false);
      detectBtn.disabled = false;
      updateDetectAvailability();
    }
  });

  videoBtn.addEventListener('click', async () => {
    hideError(videoError);

    const videoFile = videoInput.files[0];
    const adFile = adInput.files[0];
    if (!videoFile || !adFile) return;

    setLoading(true);
    setStatus('tracking video...');
    videoBtn.disabled = true;
    videoResultPanel.hidden = true;

    try {
      const form = new FormData();
      form.append('video', videoFile);
      form.append('ad', adFile);
      form.append('conf', confSlider.value);
      form.append('fit_shape', fitShape.checked);
      form.append('blend_edges', blendEdges.checked);

      const res = await fetch('/api/video/composite', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Video processing failed (${res.status})`);
      }

      const data = await res.json();
      videoResult.src = data.video_url;
      videoDownloadBtn.href = data.download_url;
      videoStats.textContent = `processed ${data.frames} frame(s), composited ${data.composited_frames} frame(s), rectangle fallback used on ${data.fallback_frames} frame(s), track id ${data.track_id ?? 'none'}.`;

      videoResultPanel.hidden = false;
      videoResultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setStatus('video proof ready');
    } catch (e) {
      showError(videoError, e.message || 'Something went wrong processing the video.');
      setStatus('video error');
    } finally {
      setLoading(false);
      videoBtn.disabled = false;
      updateVideoAvailability();
    }
  });

  boxSelect.addEventListener('change', () => runComposite({ scroll: false }));
  fitShape.addEventListener('change', () => runComposite({ scroll: false }));
  blendEdges.addEventListener('change', () => runComposite({ scroll: false }));

  async function runComposite({ scroll }) {
    hideError(compositeError);

    const sceneFile = sceneInput.files[0];
    const adFile = adInput.files[0];
    const chosen = boxes[Number(boxSelect.value)];
    if (!sceneFile || !adFile || !chosen) return;

    const token = ++compositeToken;
    setLoading(true);
    setStatus('compositing...');

    try {
      const form = new FormData();
      form.append('scene', sceneFile);
      form.append('ad', adFile);
      form.append('x1', chosen.x1);
      form.append('y1', chosen.y1);
      form.append('x2', chosen.x2);
      form.append('y2', chosen.y2);
      form.append('fit_shape', fitShape.checked);
      form.append('blend_edges', blendEdges.checked);

      if (fitShape.checked && chosen.quad) {
        form.append('quad_json', JSON.stringify(chosen.quad));
      }

      const res = await fetch('/api/composite', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Compositing failed (${res.status})`);
      }

      const data = await res.json();
      if (token !== compositeToken) return;

      resultPreview.src = data.preview;
      resultFinal.src = data.result;
      downloadBtn.href = data.result;
      fallbackNote.hidden = !(fitShape.checked && data.used_fallback);
      resultPanel.hidden = false;
      setStatus('proof ready');

      if (scroll) {
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (e) {
      if (token !== compositeToken) return;
      showError(compositeError, e.message || 'Something went wrong compositing the ad.');
      setStatus('composite error');
    } finally {
      if (token === compositeToken) setLoading(false);
    }
  }
})();