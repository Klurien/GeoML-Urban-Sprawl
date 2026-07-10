import os
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
import base64
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from api.database import get_db, init_db, Base, engine
from api.models import Analysis
from api.schemas import AnalysisResponse, AnalysisHistory
from utils.llm_client import generate_report
from utils.ml_client import segment_buildings

app = FastAPI(title="GeoML Urban Sprawl Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN", "")

@app.on_event("startup")
async def startup():
    await init_db()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GeoML Urban Sprawl Analyzer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; font-family: system-ui, sans-serif; }
            body { background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem; }
            .container { max-width: 900px; width: 100%; }
            h1 { font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .subtitle { color: #94a3b8; margin-bottom: 2rem; }
            .card { background: #1e293b; border-radius: 1rem; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
            .upload-zone { border: 2px dashed #475569; border-radius: 0.75rem; padding: 3rem; text-align: center; cursor: pointer; transition: all 0.3s; }
            .upload-zone:hover { border-color: #38bdf8; background: #1e293b; }
            .upload-zone.dragover { border-color: #38bdf8; background: rgba(56,189,248,0.1); }
            .upload-icon { font-size: 3rem; margin-bottom: 1rem; }
            .btn { background: linear-gradient(135deg, #38bdf8, #818cf8); color: #fff; border: none; padding: 0.75rem 2rem; border-radius: 0.5rem; font-size: 1rem; cursor: pointer; transition: opacity 0.3s; margin-top: 1rem; }
            .btn:hover { opacity: 0.9; }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; }
            .results { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem; }
            .results img { width: 100%; border-radius: 0.5rem; border: 1px solid #334155; }
            .stats { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
            .stat { background: #0f172a; padding: 0.75rem 1.25rem; border-radius: 0.5rem; flex: 1; min-width: 120px; text-align: center; }
            .stat-value { font-size: 1.5rem; font-weight: bold; color: #38bdf8; }
            .stat-label { font-size: 0.8rem; color: #94a3b8; }
            .report-box { background: #0f172a; border-radius: 0.5rem; padding: 1.25rem; margin-top: 1rem; border-left: 3px solid #38bdf8; line-height: 1.6; }
            .spinner { display: none; width: 40px; height: 40px; border: 3px solid #334155; border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 1rem auto; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .history-btn { background: transparent; color: #94a3b8; border: 1px solid #334155; padding: 0.5rem 1rem; border-radius: 0.5rem; cursor: pointer; font-size: 0.9rem; }
            .history-btn:hover { border-color: #38bdf8; color: #38bdf8; }
            #historyList { margin-top: 1rem; }
            .history-item { padding: 0.75rem; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; font-size: 0.9rem; }
            @media (max-width: 640px) { .results { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏙️ GeoML Urban Sprawl Analyzer</h1>
            <p class="subtitle">Upload a satellite image to detect urban areas and generate an AI policy report</p>

            <div class="card">
                <div class="upload-zone" id="dropZone">
                    <div class="upload-icon">📸</div>
                    <p>Drop a satellite image here or click to browse</p>
                    <p style="font-size:0.85rem;color:#64748b;margin-top:0.5rem;">PNG, JPG up to 10MB</p>
                    <input type="file" id="fileInput" accept="image/*" style="display:none">
                </div>
                <div style="text-align:center;">
                    <button class="btn" id="analyzeBtn" disabled>🔍 Analyze Image</button>
                </div>
                <div class="spinner" id="spinner"></div>
            </div>

            <div id="results" style="display:none;">
                <div class="card">
                    <h3>📊 Analysis Results</h3>
                    <div class="stats" id="stats"></div>
                    <div class="results">
                        <div><p style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.5rem;">Original Image</p><img id="originalImg"></div>
                        <div><p style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.5rem;">Building Detection Mask</p><img id="maskImg"></div>
                    </div>
                    <div class="report-box" id="reportBox"></div>
                </div>
            </div>

            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h3>📜 Analysis History</h3>
                    <button class="history-btn" id="historyBtn">Refresh</button>
                </div>
                <div id="historyList"><p style="color:#64748b;">No analyses yet. Upload an image to get started.</p></div>
            </div>
        </div>

        <script>
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const spinner = document.getElementById('spinner');
            const results = document.getElementById('results');
            const originalImg = document.getElementById('originalImg');
            const maskImg = document.getElementById('maskImg');
            const stats = document.getElementById('stats');
            const reportBox = document.getElementById('reportBox');
            const historyList = document.getElementById('historyList');
            const historyBtn = document.getElementById('historyBtn');

            let selectedFile = null;

            dropZone.addEventListener('click', () => fileInput.click());
            dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
            });
            fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

            function handleFile(file) {
                if (!file.type.startsWith('image/')) return alert('Please upload an image file.');
                selectedFile = file;
                analyzeBtn.disabled = false;
                dropZone.querySelector('p').textContent = `📎 ${file.name}`;
            }

            analyzeBtn.addEventListener('click', async () => {
                if (!selectedFile) return;
                analyzeBtn.disabled = true;
                spinner.style.display = 'block';
                results.style.display = 'none';

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
                    const data = await res.json();
                    displayResults(data);
                } catch (err) {
                    alert('Analysis failed: ' + err.message);
                } finally {
                    analyzeBtn.disabled = false;
                    spinner.style.display = 'none';
                }
            });

            function displayResults(data) {
                originalImg.src = data.original_image_url;
                maskImg.src = data.mask_image_url;
                stats.innerHTML = `
                    <div class="stat"><div class="stat-value">${data.urban_percentage}%</div><div class="stat-label">Urban Area</div></div>
                    <div class="stat"><div class="stat-value">${data.building_pixel_count.toLocaleString()}</div><div class="stat-label">Building Pixels</div></div>
                    <div class="stat"><div class="stat-value">${data.total_pixels.toLocaleString()}</div><div class="stat-label">Total Pixels</div></div>
                `;
                reportBox.innerHTML = data.llm_report ? `<strong>🤖 AI Policy Advisory</strong><br><br>${data.llm_report}` : '<em>No report generated.</em>';
                results.style.display = 'block';
                loadHistory();
            }

            async function loadHistory() {
                try {
                    const res = await fetch('/api/history');
                    const data = await res.json();
                    if (data.analyses.length === 0) {
                        historyList.innerHTML = '<p style="color:#64748b;">No analyses yet.</p>';
                        return;
                    }
                    historyList.innerHTML = data.analyses.slice(0, 10).map(a => `
                        <div class="history-item">
                            <span>${a.image_filename}</span>
                            <span style="color:#38bdf8;">${a.urban_percentage}% urban</span>
                            <span style="font-size:0.8rem;color:#64748b;">${new Date(a.created_at).toLocaleString()}</span>
                        </div>
                    `).join('');
                } catch (e) { /* silent */ }
            }

            historyBtn.addEventListener('click', loadHistory);
            loadHistory();
        </script>
    </body>
    </html>
    """

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported")

    contents = await file.read()

    # Run ML inference via HF Inference API directly
    try:
        ml_data = segment_buildings(contents)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(502, f"ML inference failed: {str(e)[:200]}\n{tb[:2000]}")
    if "error" in ml_data:
        raise HTTPException(502, f"ML inference error: {ml_data['error']}")

    # Generate LLM report
    report = generate_report(ml_data.get("building_pixels", 0), ml_data.get("urban_percentage", 0))

    # Store in PostgreSQL
    analysis = Analysis(
        image_filename=file.filename or "upload.png",
        building_pixel_count=ml_data.get("building_pixels", 0),
        total_pixels=ml_data.get("total_pixels", 0),
        urban_percentage=ml_data.get("urban_percentage", 0),
        llm_report=report,
        metadata_json=ml_data
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    return {
        "id": str(analysis.id),
        "image_filename": analysis.image_filename,
        "building_pixel_count": analysis.building_pixel_count,
        "total_pixels": analysis.total_pixels,
        "urban_percentage": analysis.urban_percentage,
        "llm_report": report,
        "original_image_url": f"data:image/{file.filename.split('.')[-1] if '.' in file.filename else 'png'};base64,{base64.b64encode(contents).decode()}",
        "mask_image_url": f"data:image/png;base64,{ml_data.get('mask_b64', '')}",
        "created_at": analysis.created_at.isoformat(),
    }

@app.get("/api/history")
async def get_history(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select, desc
    result = await db.execute(
        select(Analysis).order_by(desc(Analysis.created_at)).limit(20)
    )
    analyses = result.scalars().all()
    return AnalysisHistory(
        analyses=[AnalysisResponse.model_validate(a) for a in analyses],
        total=len(analyses)
    )
