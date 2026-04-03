"""FastAPI web interface for CHAOS multi-organ MRI segmentation.

Endpoints:
    GET  /          — drag-and-drop HTML frontend
    GET  /health    — health check
    POST /predict   — accepts ZIP of DICOM series, returns prediction grid

Usage:
    cd api
    uvicorn main:app --reload --port 8000
    open http://localhost:8000
"""

import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from predict import load_model, run_prediction

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="CHAOS Multi-Organ Segmentation")

CHECKPOINT = os.environ.get(
    "CHECKPOINT_PATH",
    str(Path(__file__).resolve().parent.parent / "results" / "best_model.pth")
)


@app.on_event("startup")
def startup():
    load_model(CHECKPOINT)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CHECKPOINT}


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400,
                            detail="Please upload a ZIP file containing DICOM slices.")
    try:
        zip_bytes = await file.read()
        result    = run_prediction(zip_bytes)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CHAOS — Multi-Organ MRI Segmentation</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e0e0e0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }

    h1 {
      font-size: 1.8rem;
      font-weight: 600;
      margin-bottom: 6px;
      color: #ffffff;
    }

    .subtitle {
      font-size: 0.95rem;
      color: #888;
      margin-bottom: 36px;
      text-align: center;
    }

    .legend {
      display: flex;
      gap: 18px;
      margin-bottom: 28px;
      flex-wrap: wrap;
      justify-content: center;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.85rem;
      color: #ccc;
    }

    .legend-dot {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      flex-shrink: 0;
    }

    .drop-zone {
      width: 100%;
      max-width: 520px;
      border: 2px dashed #444;
      border-radius: 14px;
      padding: 48px 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      background: #181c27;
    }

    .drop-zone:hover, .drop-zone.drag-over {
      border-color: #5b8dee;
      background: #1c2237;
    }

    .drop-zone p { color: #aaa; font-size: 0.95rem; margin-top: 10px; }
    .drop-zone .icon { font-size: 2.5rem; }

    #file-input { display: none; }

    .btn {
      margin-top: 20px;
      padding: 12px 32px;
      background: #5b8dee;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn:hover { background: #4a7de0; }
    .btn:disabled { background: #333; color: #666; cursor: not-allowed; }

    #status {
      margin-top: 20px;
      font-size: 0.9rem;
      color: #aaa;
      min-height: 22px;
    }

    #results {
      margin-top: 32px;
      width: 100%;
      max-width: 1100px;
    }

    .organs-box {
      background: #181c27;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 20px;
      font-size: 0.92rem;
    }

    .organs-box span {
      display: inline-block;
      background: #2a2f45;
      border-radius: 6px;
      padding: 3px 10px;
      margin: 3px 4px;
      color: #ddd;
    }

    #grid-img {
      width: 100%;
      border-radius: 10px;
      border: 1px solid #2a2f45;
    }

    .error {
      color: #ff6b6b;
      background: #2a1a1a;
      border-radius: 8px;
      padding: 12px 18px;
      margin-top: 16px;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>

  <h1>Multi-Organ MRI Segmentation</h1>
  <p class="subtitle">Upload a ZIP of T2SPIR DICOM slices — the model segments liver,
    kidneys and spleen.</p>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-dot" style="background:#ff5050"></div> Liver
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#50b450"></div> Right kidney
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#5050ff"></div> Left kidney
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#ffc850"></div> Spleen
    </div>
  </div>

  <div class="drop-zone" id="drop-zone">
    <div class="icon">🗂️</div>
    <p>Drag &amp; drop a <strong>.zip</strong> of DICOM files here</p>
    <p>or click to browse</p>
    <input type="file" id="file-input" accept=".zip"/>
  </div>

  <button class="btn" id="run-btn" disabled>Run segmentation</button>
  <div id="status"></div>

  <div id="results" style="display:none">
    <div class="organs-box" id="organs-box"></div>
    <img id="grid-img" alt="Prediction grid"/>
  </div>

  <script>
    const dropZone  = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const runBtn    = document.getElementById('run-btn');
    const status    = document.getElementById('status');
    const results   = document.getElementById('results');
    const organsBox = document.getElementById('organs-box');
    const gridImg   = document.getElementById('grid-img');

    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', e => {
      e.preventDefault(); dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f && f.name.endsWith('.zip')) setFile(f);
      else showError('Please drop a .zip file.');
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) setFile(fileInput.files[0]);
    });

    function setFile(f) {
      selectedFile = f;
      dropZone.querySelector('p').textContent = `Selected: ${f.name}`;
      runBtn.disabled = false;
      results.style.display = 'none';
      status.textContent = '';
    }

    runBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      runBtn.disabled = true;
      status.textContent = 'Running segmentation…';
      results.style.display = 'none';

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const resp = await fetch('/predict', { method: 'POST', body: formData });
        const data = await resp.json();

        if (!resp.ok) {
          showError(data.detail || 'Server error');
          return;
        }

        // Show organs found
        organsBox.innerHTML = '<strong>Organs detected:</strong> '
          + data.organs_found.map(o => `<span>${o}</span>`).join('')
          + `&nbsp;&nbsp;<span style="color:#888">${data.n_slices} slices processed</span>`;

        // Show grid
        gridImg.src = 'data:image/png;base64,' + data.grid_png;
        results.style.display = 'block';
        status.textContent = 'Done.';
      } catch (err) {
        showError('Request failed: ' + err.message);
      } finally {
        runBtn.disabled = false;
      }
    });

    function showError(msg) {
      status.innerHTML = `<div class="error">${msg}</div>`;
    }
  </script>
</body>
</html>
"""
