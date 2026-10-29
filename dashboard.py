"""
Facade-Renderer: Interactive Web Dashboard
A lightweight, modern web interface for testing and running the Pix2Pix Facade Generator.
Powered by Flask (zero extra heavy dependencies needed).
"""

import os
import re
import io
import time
import glob
import math
import base64
import argparse
from typing import Optional, Tuple

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from flask import Flask, request, jsonify, render_template_string

from models import UNetGenerator
from dataset import find_facades_dir

app = Flask(__name__)

# Global model state
CURRENT_MODEL: Optional[UNetGenerator] = None
CURRENT_CHECKPOINT_PATH: Optional[str] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACADES_DIR = find_facades_dir() or r"D:\Datasets\facades"


def natural_sort_key(s: str):
    """Sort strings containing numbers in natural numeric order (e.g. 1.jpg, 2.jpg ... 10.jpg)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_generator(checkpoint_path: str) -> UNetGenerator:
    """Loads and caches the generator model from checkpoint."""
    global CURRENT_MODEL, CURRENT_CHECKPOINT_PATH
    if CURRENT_MODEL is not None and CURRENT_CHECKPOINT_PATH == checkpoint_path:
        return CURRENT_MODEL

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = UNetGenerator(in_channels=3, out_channels=3).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    CURRENT_MODEL = model
    CURRENT_CHECKPOINT_PATH = checkpoint_path
    print(f"[Dashboard] Loaded model from: {checkpoint_path} on {DEVICE}")
    return CURRENT_MODEL


def pil_to_base64(img: Image.Image) -> str:
    """Converts PIL image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_inference(model: UNetGenerator, condition_pil: Image.Image) -> Tuple[Image.Image, float]:
    """Runs generator inference on a PIL condition image, returning (generated_pil, latency_ms)."""
    resized = TF.resize(condition_pil, [256, 256], interpolation=T.InterpolationMode.BICUBIC)
    tensor = TF.to_tensor(resized)
    normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tensor = normalize(tensor).unsqueeze(0).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        fake_tensor = model(tensor)
    latency_ms = (time.time() - t0) * 1000

    # Denormalize to [0, 1]
    denorm = torch.clamp((fake_tensor[0].cpu() + 1.0) / 2.0, 0.0, 1.0)
    fake_pil = TF.to_pil_image(denorm)
    return fake_pil, latency_ms


def compute_metrics(pred_pil: Image.Image, gt_pil: Image.Image) -> Tuple[float, float]:
    """Calculates L1 error and PSNR between prediction and ground truth."""
    pred_res = TF.resize(pred_pil, [256, 256], interpolation=T.InterpolationMode.BICUBIC)
    gt_res = TF.resize(gt_pil, [256, 256], interpolation=T.InterpolationMode.BICUBIC)

    p_tensor = TF.to_tensor(pred_res)
    g_tensor = TF.to_tensor(gt_res)

    l1 = torch.mean(torch.abs(p_tensor - g_tensor)).item()
    mse = torch.mean((p_tensor - g_tensor) ** 2).item()
    psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 99.0
    return round(l1, 4), round(psnr, 2)


# HTML Dashboard Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Facade-Renderer | Interactive Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(30, 41, 59, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      font-size: 24px;
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
    }
    .brand-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    .badge {
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 9999px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
      font-weight: 500;
    }
    main {
      flex: 1;
      max-width: 1300px;
      width: 100%;
      margin: 0 auto;
      padding: 28px 20px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .grid-top {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 850px) {
      .grid-top { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .card-title {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    select, input[type="text"] {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 14px;
      outline: none;
      transition: border 0.2s;
    }
    select:focus, input[type="text"]:focus {
      border-color: var(--accent);
    }
    .tabs {
      display: flex;
      gap: 8px;
      background: #0f172a;
      padding: 4px;
      border-radius: 8px;
    }
    .tab {
      flex: 1;
      padding: 8px;
      text-align: center;
      font-size: 13px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      color: var(--text-muted);
      transition: all 0.2s;
    }
    .tab.active {
      background: var(--card-bg);
      color: var(--text);
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .upload-zone {
      border: 2px dashed var(--card-border);
      border-radius: 8px;
      padding: 28px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      background: #0f172a80;
    }
    .upload-zone:hover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.05);
    }
    .btn {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 12px 20px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: background 0.2s, transform 0.1s;
    }
    .btn:hover {
      background: var(--accent-hover);
    }
    .btn:active {
      transform: scale(0.98);
    }
    .btn-secondary {
      background: #334155;
      color: var(--text);
    }
    .btn-secondary:hover {
      background: #475569;
    }
    .viewer {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
    }
    @media (max-width: 900px) {
      .viewer { grid-template-columns: 1fr; }
    }
    .img-box {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .img-header {
      padding: 10px 14px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .img-wrap {
      aspect-ratio: 1;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #050b14;
    }
    .img-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .img-placeholder {
      color: var(--text-muted);
      font-size: 13px;
    }
    .metrics-bar {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }
    .metric {
      background: #0f172a;
      padding: 10px 16px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .metric-name {
      font-size: 11px;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 600;
    }
    .metric-val {
      font-size: 18px;
      font-weight: 700;
      color: var(--accent);
    }
    .spinner {
      display: none;
      width: 16px;
      height: 16px;
      border: 2px solid #0f172a;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="brand-icon">🏛️</span>
      <span class="brand-title">Facade-Renderer</span>
      <span class="badge" id="deviceBadge">Device: {{ device }}</span>
    </div>
    <div>
      <a href="https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset" target="_blank" style="color: var(--text-muted); font-size: 13px; text-decoration: none;">CMP Facades Dataset ↗</a>
    </div>
  </header>

  <main>
    <div class="grid-top">
      <!-- 1. Checkpoint & Configuration -->
      <div class="card">
        <div class="card-title">⚙️ Model Checkpoint</div>
        <div>
          <label style="font-size: 12px; color: var(--text-muted); display: block; margin-bottom: 6px;">Available Checkpoints:</label>
          <select id="checkpointSelect" onchange="onCheckpointChange()">
            {% for ckpt in checkpoints %}
            <option value="{{ ckpt }}">{{ ckpt }}</option>
            {% endfor %}
            <option value="__custom__">-- Custom Checkpoint Path --</option>
          </select>
        </div>
        <div id="customCkptWrap" style="display: none;">
          <label style="font-size: 12px; color: var(--text-muted); display: block; margin-bottom: 6px;">Custom Path (.pth):</label>
          <input type="text" id="customCkptInput" placeholder="path/to/generator.pth">
        </div>
        <div style="font-size: 12px; color: var(--text-muted);">
          Direction: <strong style="color: var(--text);">BtoA</strong> (Blueprint / Mask → Photorealistic Facade)
        </div>
      </div>

      <!-- 2. Test Input Selection -->
      <div class="card">
        <div class="card-title">🖼️ Input Selection</div>
        <div class="tabs">
          <div class="tab active" id="tabPreset" onclick="switchTab('preset')">Test Set Gallery</div>
          <div class="tab" id="tabCustom" onclick="switchTab('custom')">Upload Custom Blueprint</div>
        </div>

        <div id="sectionPreset">
          <label style="font-size: 12px; color: var(--text-muted); display: block; margin-bottom: 6px;">Choose sample from test split (106 samples):</label>
          <select id="testSampleSelect" onchange="loadPresetSample()">
            {% for s in test_samples %}
            <option value="{{ s }}">{{ s }}</option>
            {% endfor %}
          </select>
        </div>

        <div id="sectionCustom" style="display: none;">
          <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="handleFileUpload(event)">
            <p id="uploadText" style="font-size: 13px; color: var(--text-muted);">Click to upload blueprint (PNG/JPG)</p>
          </div>
        </div>

        <button class="btn" id="generateBtn" onclick="renderFacade()">
          <span class="spinner" id="btnSpinner"></span>
          <span>⚡ Render Facade</span>
        </button>
      </div>
    </div>

    <!-- 3. Visual Viewer Results -->
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="card-title">🔍 Visual Comparison & Evaluation</div>
        <div class="metrics-bar" id="metricsBar" style="display: none;">
          <div class="metric">
            <span class="metric-name">Inference Latency</span>
            <span class="metric-val" id="valLatency">-- ms</span>
          </div>
          <div class="metric" id="boxL1">
            <span class="metric-name">L1 MAE Error</span>
            <span class="metric-val" id="valL1">--</span>
          </div>
          <div class="metric" id="boxPSNR">
            <span class="metric-name">PSNR</span>
            <span class="metric-val" id="valPSNR">-- dB</span>
          </div>
        </div>
      </div>

      <div class="viewer">
        <!-- Panel 1: Condition Blueprint -->
        <div class="img-box">
          <div class="img-header">
            <span>INPUT: Blueprint / Label Map</span>
          </div>
          <div class="img-wrap">
            <img id="imgCondition" src="" style="display: none;">
            <span class="img-placeholder" id="phCondition">Select or upload an image</span>
          </div>
        </div>

        <!-- Panel 2: Ground Truth -->
        <div class="img-box">
          <div class="img-header">
            <span>GROUND TRUTH: Real Facade</span>
          </div>
          <div class="img-wrap">
            <img id="imgGroundTruth" src="" style="display: none;">
            <span class="img-placeholder" id="phGroundTruth">N/A for custom uploads</span>
          </div>
        </div>

        <!-- Panel 3: Generated Output -->
        <div class="img-box" style="border-color: var(--accent);">
          <div class="img-header" style="background: rgba(56, 189, 248, 0.1); color: var(--accent);">
            <span>GENERATED: Building Photo (cGAN)</span>
            <a id="downloadLink" href="#" download="facade_generated.png" style="display:none; color: var(--accent); font-size: 12px; text-decoration: none;">Download 💾</a>
          </div>
          <div class="img-wrap">
            <img id="imgGenerated" src="" style="display: none;">
            <span class="img-placeholder" id="phGenerated">Click "Render Facade"</span>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    let activeMode = 'preset';
    let currentCustomB64 = null;

    function switchTab(mode) {
      activeMode = mode;
      document.getElementById('tabPreset').classList.toggle('active', mode === 'preset');
      document.getElementById('tabCustom').classList.toggle('active', mode === 'custom');
      document.getElementById('sectionPreset').style.display = mode === 'preset' ? 'block' : 'none';
      document.getElementById('sectionCustom').style.display = mode === 'custom' ? 'block' : 'none';

      if (mode === 'preset') {
        loadPresetSample();
      } else {
        clearGroundTruth();
      }
    }

    function onCheckpointChange() {
      const val = document.getElementById('checkpointSelect').value;
      document.getElementById('customCkptWrap').style.display = (val === '__custom__') ? 'block' : 'none';
    }

    function getSelectedCheckpoint() {
      const val = document.getElementById('checkpointSelect').value;
      if (val === '__custom__') {
        return document.getElementById('customCkptInput').value.trim();
      }
      return val;
    }

    function loadPresetSample() {
      const sample = document.getElementById('testSampleSelect').value;
      if (!sample) return;

      fetch(`/api/sample_info?sample=${encodeURIComponent(sample)}`)
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            document.getElementById('imgCondition').src = data.condition_b64;
            document.getElementById('imgCondition').style.display = 'block';
            document.getElementById('phCondition').style.display = 'none';

            document.getElementById('imgGroundTruth').src = data.ground_truth_b64;
            document.getElementById('imgGroundTruth').style.display = 'block';
            document.getElementById('phGroundTruth').style.display = 'none';

            // Clear previous generated output
            document.getElementById('imgGenerated').style.display = 'none';
            document.getElementById('phGenerated').style.display = 'block';
            document.getElementById('downloadLink').style.display = 'none';
            document.getElementById('metricsBar').style.display = 'none';
          }
        });
    }

    function handleFileUpload(e) {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function(evt) {
        currentCustomB64 = evt.target.result;
        document.getElementById('imgCondition').src = currentCustomB64;
        document.getElementById('imgCondition').style.display = 'block';
        document.getElementById('phCondition').style.display = 'none';
        document.getElementById('uploadText').innerText = file.name;
        clearGroundTruth();
      };
      reader.readAsDataURL(file);
    }

    function clearGroundTruth() {
      document.getElementById('imgGroundTruth').style.display = 'none';
      document.getElementById('phGroundTruth').style.display = 'block';
      document.getElementById('phGroundTruth').innerText = 'N/A for custom uploads';
      document.getElementById('boxL1').style.display = 'none';
      document.getElementById('boxPSNR').style.display = 'none';
    }

    function renderFacade() {
      const ckpt = getSelectedCheckpoint();
      if (!ckpt) {
        alert('Please specify a valid model checkpoint path (.pth).');
        return;
      }

      const btn = document.getElementById('generateBtn');
      const spinner = document.getElementById('btnSpinner');
      btn.disabled = true;
      spinner.style.display = 'inline-block';

      const payload = {
        checkpoint: ckpt,
        mode: activeMode,
        sample: activeMode === 'preset' ? document.getElementById('testSampleSelect').value : null,
        custom_b64: activeMode === 'custom' ? currentCustomB64 : null
      };

      fetch('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(res => res.json())
      .then(data => {
        btn.disabled = false;
        spinner.style.display = 'none';

        if (data.error) {
          alert('Inference error: ' + data.error);
          return;
        }

        // Show Generated
        const genImg = document.getElementById('imgGenerated');
        genImg.src = data.generated_b64;
        genImg.style.display = 'block';
        document.getElementById('phGenerated').style.display = 'none';

        // Download link
        const dl = document.getElementById('downloadLink');
        dl.href = data.generated_b64;
        dl.style.display = 'inline';

        // Metrics
        document.getElementById('metricsBar').style.display = 'flex';
        document.getElementById('valLatency').innerText = Math.round(data.latency_ms) + ' ms';

        if (data.l1 !== null) {
          document.getElementById('boxL1').style.display = 'flex';
          document.getElementById('boxPSNR').style.display = 'flex';
          document.getElementById('valL1').innerText = data.l1;
          document.getElementById('valPSNR').innerText = data.psnr + ' dB';
        } else {
          document.getElementById('boxL1').style.display = 'none';
          document.getElementById('boxPSNR').style.display = 'none';
        }
      })
      .catch(err => {
        btn.disabled = false;
        spinner.style.display = 'none';
        alert('Network or server error: ' + err);
      });
    }

    // Initialize preset on page load
    window.onload = function() {
      onCheckpointChange();
      if (document.getElementById('testSampleSelect').options.length > 0) {
        loadPresetSample();
      }
    };
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    # Discover available checkpoints
    checkpoints = sorted(
        list(set(glob.glob("outputs/checkpoints/*.pth") + glob.glob("checkpoints/*.pth"))),
        key=natural_sort_key,
    )
    if not checkpoints:
        checkpoints = ["outputs/checkpoints/generator_latest.pth"]

    # Discover test samples in natural numeric order (1.jpg, 2.jpg ... 106.jpg)
    test_dir = os.path.join(FACADES_DIR, "test")
    test_samples = []
    if os.path.exists(test_dir):
        files = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png"))
        test_samples = sorted([os.path.basename(f) for f in files], key=natural_sort_key)

    return render_template_string(
        HTML_TEMPLATE,
        device=str(DEVICE).upper(),
        checkpoints=checkpoints,
        test_samples=test_samples,
    )


@app.route("/api/sample_info", methods=["GET"])
def sample_info():
    sample_name = request.args.get("sample", "")
    sample_path = os.path.join(FACADES_DIR, "test", sample_name)

    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": "Sample not found"}), 404

    img = Image.open(sample_path).convert("RGB")
    w, h = img.size
    half_w = w // 2
    left = img.crop((0, 0, half_w, h))    # Ground Truth photo
    right = img.crop((half_w, 0, w, h))   # Blueprint condition

    return jsonify({
        "success": True,
        "condition_b64": pil_to_base64(right),
        "ground_truth_b64": pil_to_base64(left),
    })


@app.route("/api/render", methods=["POST"])
def render_endpoint():
    data = request.json or {}
    checkpoint = data.get("checkpoint")
    mode = data.get("mode", "preset")

    if not checkpoint:
        return jsonify({"error": "No checkpoint specified"}), 400

    try:
        model = load_generator(checkpoint)
    except Exception as e:
        return jsonify({"error": f"Failed to load checkpoint: {str(e)}"}), 500

    gt_pil = None
    if mode == "preset":
        sample_name = data.get("sample", "")
        sample_path = os.path.join(FACADES_DIR, "test", sample_name)
        if not os.path.exists(sample_path):
            return jsonify({"error": f"Preset sample not found: {sample_name}"}), 404
        img = Image.open(sample_path).convert("RGB")
        w, h = img.size
        gt_pil = img.crop((0, 0, w // 2, h))
        cond_pil = img.crop((w // 2, 0, w, h))
    else:
        custom_b64 = data.get("custom_b64")
        if not custom_b64:
            return jsonify({"error": "No custom image uploaded"}), 400
        header, encoded = custom_b64.split(",", 1) if "," in custom_b64 else ("", custom_b64)
        img_bytes = base64.b64decode(encoded)
        cond_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    try:
        generated_pil, latency = run_inference(model, cond_pil)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    l1_val, psnr_val = None, None
    if gt_pil is not None:
        l1_val, psnr_val = compute_metrics(generated_pil, gt_pil)

    return jsonify({
        "success": True,
        "generated_b64": pil_to_base64(generated_pil),
        "latency_ms": latency,
        "l1": l1_val,
        "psnr": psnr_val,
    })


def main():
    parser = argparse.ArgumentParser(description="Facade-Renderer Interactive Testing Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pre-load a specific checkpoint")
    args = parser.parse_args()

    if args.checkpoint and os.path.exists(args.checkpoint):
        load_generator(args.checkpoint)

    print("=" * 60)
    print("Facade-Renderer: Interactive Testing Dashboard")
    print(f"Server running at: http://{args.host}:{args.port}")
    print(f"Device:           {DEVICE}")
    print(f"Dataset location: {FACADES_DIR}")
    print("=" * 60)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
