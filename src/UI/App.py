import os, json, math, pickle, io
import numpy as np
import requests as req
from flask import Flask, render_template_string, request, jsonify
from PIL import Image

CLIP_AVAILABLE = False
clip_model = clip_processor = clip_text_features = clip_device = None
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    _POSITIVE = [
        "a modern renovated apartment with new furniture and clean walls",
        "a bright newly furnished flat with modern kitchen and bathroom",
        "a stylish contemporary apartment with fresh renovation",
        "a clean well-maintained apartment interior",
        "a luxury apartment with high-end finishes and modern design",
    ]
    _NEGATIVE = [
        "an old rundown apartment with damaged walls and old furniture",
        "a dirty neglected flat with peeling paint and worn floors",
        "an outdated apartment with old wallpaper and broken fixtures",
        "a dark shabby apartment in poor condition",
        "an empty unfurnished apartment with bare concrete walls",
    ]
    _N_POS = len(_POSITIVE)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(clip_device)
    clip_model.eval()

    _inputs = clip_processor(text=_POSITIVE + _NEGATIVE,
                             return_tensors="pt", padding=True,
                             truncation=True).to(clip_device)
    with torch.no_grad():
        _tf = clip_model.get_text_features(**_inputs)
        _tf = _tf if isinstance(_tf, torch.Tensor) else _tf.pooler_output
        clip_text_features = _tf / _tf.norm(dim=-1, keepdim=True)

    CLIP_AVAILABLE = True
except Exception as e:
    pass


def score_photo(img_bytes: bytes) -> float:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt",
                            padding=True).to(clip_device)
    with torch.no_grad():
        out = clip_model.get_image_features(**inputs)
        imf = out if isinstance(out, torch.Tensor) else out.pooler_output
        imf = imf / imf.norm(dim=-1, keepdim=True)
    sims = (imf @ clip_text_features.T).squeeze(0).cpu().numpy()
    pos_score = float(sims[:_N_POS].mean())
    neg_score = float(sims[_N_POS:].mean())
    raw = pos_score - neg_score
    score = (raw + 0.06) / 0.12
    return round(max(0.0, min(1.0, score)), 4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "model1.pkl")
POI_PATH = os.path.join(PROJECT_ROOT, "data", "poi_czech_republic.json")
CITY_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "city_price_index1.json")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
input_features = bundle["input_features"]
poi_medians = bundle["poi_medians"]
cities = bundle.get("city_index", {}).get("cities", {})
global_cpm = bundle.get("city_index", {}).get("global_mean", 372.0)
city_disp_meds = bundle.get("city_disp_medians", {})
disp_meds = bundle.get("disp_medians", {})
t1_rel = bundle.get("t1_rel", 0.800)
t2_rel = bundle.get("t2_rel", 1.086)
has_photo = bundle.get("has_photo", False)
le_mesto = bundle.get("le_mesto")
importances = bundle.get("importances", {})

poi_data = None
if os.path.exists(POI_PATH):
    with open(POI_PATH, encoding="utf-8") as f:
        poi_data = json.load(f)


def nearest_poi(lat, lon, bucket):
    lats, lons = bucket["lat"], bucket["lon"]
    if not lats:
        return None
    R, lr, mn = 6_371_000, math.radians(lat), float("inf")
    for pl, pn in zip(lats, lons):
        if abs(pl - lat) > 0.05 or abs(pn - lon) > 0.05:
            continue
        dp = math.radians(pl - lat)
        dl = math.radians(pn - lon)
        a = math.sin(dp / 2) ** 2 + math.cos(lr) * math.cos(math.radians(pl)) * math.sin(dl / 2) ** 2
        d = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        if d < mn:
            mn = d
    return round(mn, 1) if mn != float("inf") else None


def get_poi_from_gps(lat, lon):
    keys = ["skola_m", "park_m", "mhd_m", "lekarna_m", "supermarket_m"]
    return {
        k: (nearest_poi(lat, lon, poi_data[k])
            if poi_data and k in poi_data and nearest_poi(lat, lon, poi_data[k]) is not None
            else poi_medians.get(k, 500))
        for k in keys
    }


def geocode(addr):
    try:
        r = req.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": addr + ", Česká republika", "format": "json", "limit": 1},
            headers={"User-Agent": "RentalPredictor/3.0"},
            timeout=10,
        )
        if r.status_code == 200 and r.json():
            res = r.json()[0]
            return float(res["lat"]), float(res["lon"]), res.get("display_name", "")
    except:
        pass
    return None, None, None


def find_city(resolved):
    if not resolved:
        return None
    parts = [x.strip() for x in resolved.split(",")]
    known = set(cities.keys()) | {k.split("|")[0] for k in city_disp_meds.keys()}
    sorted_known = sorted(known, key=len, reverse=True)
    for p in parts:
        if p in known:
            return p
    for c in sorted_known:
        for p in parts:
            if c == p or c in p:
                return c
    return None


def encode_mesto(city):
    if le_mesto is None:
        return 0
    try:
        return int(le_mesto.transform([city or "neznámo"])[0])
    except ValueError:
        return 0


def get_ref_median(city, disp):
    key = f"{city}|{disp}" if city else None
    if key and key in city_disp_meds:
        return city_disp_meds[key], city
    return disp_meds.get(int(disp), 15000), "ČR"


app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="cs">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cenové pásmo pronájmu</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0f1a;--card:#111827;--border:#1e293b;--text:#e2e8f0;--dim:#94a3b8;
  --accent:#3b82f6;--glow:rgba(59,130,246,.15);
  --green:#10b981;--amber:#f59e0b;--red:#ef4444;--r:12px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);
  min-height:100vh;display:flex;justify-content:center;padding:32px 16px}
.wrap{width:100%;max-width:580px}
h1{font-size:1.45rem;font-weight:700;text-align:center;margin-bottom:4px}
.sub{color:var(--dim);font-size:.82rem;text-align:center;margin-bottom:24px;line-height:1.5}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:22px;margin-bottom:12px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px}.full{grid-column:1/-1}
label{display:block;font-size:.7rem;font-weight:500;color:var(--dim);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
input,select{width:100%;padding:9px 12px;background:var(--bg);border:1px solid var(--border);
  border-radius:7px;color:var(--text);font-family:'DM Mono',monospace;font-size:.88rem;outline:none;transition:border .2s}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--glow)}
input[type=range]{padding:4px 0;cursor:pointer}
input::placeholder{color:#475569}select option{background:var(--card)}
.btn{width:100%;padding:12px;margin-top:14px;background:var(--accent);color:#fff;
  font-family:'DM Sans',sans-serif;font-size:.92rem;font-weight:600;border:none;border-radius:8px;cursor:pointer;transition:all .2s}
.btn:hover{background:#2563eb;transform:translateY(-1px)}.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.geo{font-size:.74rem;color:var(--dim);margin-top:4px;min-height:1em}
.poi-sim{margin-top:16px;padding-top:16px;border-top:1px solid var(--border)}
.poi-sim-title{font-size:.72rem;font-weight:500;color:var(--dim);text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}
.poi-row{display:grid;grid-template-columns:120px 1fr 56px;align-items:center;gap:8px;margin-bottom:8px;font-size:.8rem}
.poi-lbl{color:var(--dim)}.poi-val{color:var(--text);font-family:'DM Mono',monospace;font-size:.75rem;text-align:right}
.photo-upload{margin-top:16px;padding-top:16px;border-top:1px solid var(--border)}
.photo-upload-title{font-size:.72rem;font-weight:500;color:var(--dim);text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}
.photo-drop{border:1px dashed var(--border);border-radius:8px;padding:16px;text-align:center;cursor:pointer;transition:border .2s;position:relative}
.photo-drop:hover{border-color:var(--accent)}
.photo-drop input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.photo-drop .ph-hint{font-size:.78rem;color:var(--dim)}
.photo-drop .ph-name{font-size:.78rem;color:var(--text);margin-top:4px}
.photo-score-row{display:flex;align-items:center;gap:10px;margin-top:10px;font-size:.8rem}
.photo-score-lbl{color:var(--dim);width:100px}
.photo-score-bar{flex:1;height:6px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden}
.photo-score-fill{height:100%;border-radius:3px;transition:width .5s;background:var(--accent)}
.photo-score-val{color:var(--text);font-family:'DM Mono',monospace;font-size:.76rem;min-width:42px;text-align:right}

@keyframes fi{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
.tc{text-align:center;padding-bottom:16px}
.tl{font-size:.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.badge{display:inline-block;padding:7px 32px;border-radius:999px;font-size:1.35rem;font-weight:700;margin-bottom:6px}
.b0{background:rgba(16,185,129,.12);color:var(--green);border:1px solid rgba(16,185,129,.3)}
.b1{background:rgba(245,158,11,.12);color:var(--amber);border:1px solid rgba(245,158,11,.3)}
.b2{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.ctx{margin:0 0 16px;padding:14px 16px;border-radius:9px;font-size:.84rem;line-height:1.65}
.ctx0{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2)}
.ctx1{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2)}
.ctx2{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.2)}
.ctx strong{font-weight:600}
.ctx .note{font-size:.75rem;color:var(--dim);margin-top:7px;line-height:1.5}
.ruler-wrap{margin:0 0 16px}
.rtitle{font-size:.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:9px}
.ruler{position:relative;height:9px;border-radius:5px;
  background:linear-gradient(90deg,var(--green) 0%,var(--green) 33%,var(--amber) 33%,var(--amber) 66%,var(--red) 66%,var(--red) 100%)}
.rmark{position:absolute;top:-5px;width:19px;height:19px;border-radius:50%;background:#fff;border:3px solid #0a0f1a;transform:translateX(-50%);transition:left .5s ease;box-shadow:0 0 0 2px var(--accent)}
.rlbls{display:flex;justify-content:space-between;font-size:.68rem;color:var(--dim);font-family:'DM Mono',monospace;margin-top:5px}
.zones{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:9px;text-align:center}
.zone{padding:8px 4px;border-radius:6px;font-size:.73rem}
.z0{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.18);color:var(--green)}
.z1{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.18);color:var(--amber)}
.z2{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.18);color:var(--red)}
.zone .zt{font-weight:600;font-size:.78rem;font-family:'DM Mono',monospace}
.zone .zb{color:var(--dim);margin-top:2px;font-size:.68rem}
.zone.active{border-width:2px;background-color:rgba(255,255,255,.04)}
.cbar{display:flex;align-items:center;gap:9px;margin-bottom:6px;font-size:.79rem}
.clbl{width:58px;color:var(--dim)}.cbg{flex:1;height:6px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden}
.cfill{height:100%;border-radius:3px;transition:width .6s ease}
.cval{width:32px;text-align:right;color:var(--dim);font-family:'DM Mono',monospace;font-size:.76rem}
.dg{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:12px}
.di{background:var(--bg);border-radius:7px;padding:8px 10px}
.dl{font-size:.65rem;color:var(--dim);text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px}
.dv{font-family:'DM Mono',monospace;font-size:.82rem}
.info{background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.15);border-radius:var(--r);
  padding:14px 16px;font-size:.78rem;color:var(--dim);line-height:1.65;margin-top:4px}
.info strong{color:var(--text)}
</style>
</head>
<body>
<div class="wrap">
<h1>Cenové pásmo pronájmu</h1>
<p class="sub">Zjistěte, zda je váš byt levný, průměrný nebo drahý<br>oproti srovnatelným bytům ve stejné lokalitě.</p>

<div class="card">
  <div class="g2">
    <div class="full">
      <label>Adresa bytu</label>
      <input id="adr" type="text" placeholder="Např. Náměstí Míru 1, Praha 2">
      <div class="geo" id="geo"></div>
    </div>
    <div><label>Plocha (m²)</label><input id="pl" type="number" min="10" max="500" placeholder="65"></div>
    <div>
      <label>Dispozice</label>
      <select id="di">
        <option value="1">Garsoniéra</option>
        <option value="2">1+kk / 1+1</option>
        <option value="3" selected>2+kk / 2+1</option>
        <option value="4">3+kk / 3+1</option>
        <option value="5">4+kk a více</option>
      </select>
    </div>
  </div>
  <div class="poi-sim">
    <div class="poi-sim-title">Vzdálenosti k POI – automaticky z GPS, lze upravit</div>    <div class="poi-row">
      <span class="poi-lbl">MHD zastávka</span>
      <input type="range" id="s_mhd" min="0" max="2000" step="50" value="300" oninput="updateSlider('mhd',this.value)">
      <span class="poi-val" id="v_mhd">300 m</span>
    </div>
    <div class="poi-row">
      <span class="poi-lbl">Škola</span>
      <input type="range" id="s_skola" min="0" max="3000" step="50" value="500" oninput="updateSlider('skola',this.value)">
      <span class="poi-val" id="v_skola">500 m</span>
    </div>
    <div class="poi-row">
      <span class="poi-lbl">Park</span>
      <input type="range" id="s_park" min="0" max="3000" step="50" value="400" oninput="updateSlider('park',this.value)">
      <span class="poi-val" id="v_park">400 m</span>
    </div>
    <div class="poi-row">
      <span class="poi-lbl">Lékárna</span>
      <input type="range" id="s_lekarna" min="0" max="3000" step="50" value="500" oninput="updateSlider('lekarna',this.value)">
      <span class="poi-val" id="v_lekarna">500 m</span>
    </div>
    <div class="poi-row">
      <span class="poi-lbl">Supermarket</span>
      <input type="range" id="s_super" min="0" max="2000" step="50" value="300" oninput="updateSlider('super',this.value)">
      <span class="poi-val" id="v_super">300 m</span>
    </div>
  </div>
  {% if clip_available %}
  <div class="photo-upload">
    <div class="photo-upload-title">Fotka bytu – pro přesnější výsledek (volitelné)</div>
    <div class="photo-drop" onclick="document.getElementById('photo-input').click()">
      <input type="file" id="photo-input" accept="image/*" onchange="uploadPhoto(this)">
      <div class="ph-hint">Klikni nebo přetáhni fotku interiéru</div>
      <div class="ph-name" id="photo-name"></div>
    </div>
    <div class="photo-score-row" id="photo-score-row" style="display:none">
      <span class="photo-score-lbl">Kvalita (CLIP)</span>
      <div class="photo-score-bar"><div class="photo-score-fill" id="photo-score-fill" style="width:0%"></div></div>
      <span class="photo-score-val" id="photo-score-val">—</span>
    </div>
  </div>
  {% endif %}
  <button class="btn" onclick="go()">Zařadit do cenového pásma</button>
</div>

<div id="result">
<div class="card">
  <div class="tc">
    <div class="tl">Cenové pásmo</div>
    <div id="badge" class="badge">—</div>
    <div style="font-size:.8rem;color:var(--dim)" id="conf"></div>
  </div>
  <div id="ctx" class="ctx"></div>
  <div class="ruler-wrap">
    <div class="rtitle">Poloha v cenovém spektru (<span id="dname"></span>)</div>
    <div class="ruler"><div class="rmark" id="mk"></div></div>
    <div class="rlbls"><span>0</span><span id="rl1"></span><span id="rl2"></span><span id="rlmax"></span></div>
    <div class="zones" id="zones"></div>
  </div>
  <div style="margin-bottom:4px">
    <div style="font-size:.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:7px">Pravděpodobnost pásem</div>
    <div id="cbars"></div>
  </div>
  <div class="dg" id="dg"></div>
</div>
</div>

<div class="info">
  <strong>Jak funguje klasifikace?</strong><br>
  Model porovnává byt s <em>mediánem cen pro stejnou dispozici v daném městě</em>.
  Výsledek říká: <em>„Je tento byt levný, průměrný nebo drahý?"</em><br><br>
  <strong>Vzdálenosti k POI</strong> se automaticky vypočítají z GPS adresy.
  Pokud znáte přesnější hodnoty, upravte posuvníky.<br><br>
  Natrénováno na {{ n_data }} inzerátech ze Sreality.
  {% if has_photo %}Zahrnuje hodnocení kvality fotek (CLIP).{% endif %}
</div>
</div>

<script>
const T=[{l:"Levné",c:"b0",col:"#10b981"},{l:"Střední",c:"b1",col:"#f59e0b"},{l:"Drahé",c:"b2",col:"#ef4444"}];
const fmt = n => Math.round(n).toLocaleString('cs-CZ') + ' Kč';
const poi = {mhd:null, skola:null, park:null, lekarna:null, super:null};
let poiUserModified = false;
let currentPhotoScore = 0.5;

function updateSlider(key, val) {
  poi[key] = parseInt(val);
  document.getElementById('v_' + key).textContent = val + ' m';
  poiUserModified = true;
}

function setSliders(poiData) {
  const map = {mhd:'mhd_m', skola:'skola_m', park:'park_m', lekarna:'lekarna_m', super:'supermarket_m'};
  for (const [k, col] of Object.entries(map)) {
    const v = Math.round(poiData[col]);
    const max = parseInt(document.getElementById('s_' + k).max);
    document.getElementById('s_' + k).value = Math.min(v, max);
    document.getElementById('v_' + k).textContent = v + ' m';
    poi[k] = v;
  }
  poiUserModified = false;
}

async function uploadPhoto(input) {
  if (!input.files[0]) return;
  const file = input.files[0];
  document.getElementById('photo-name').textContent = file.name;
  document.getElementById('photo-score-row').style.display = 'none';

  const fd = new FormData();
  fd.append('photo', file);

  try {
    const r = await fetch('/score_photo', {method:'POST', body:fd});
    const d = await r.json();
    if (d.error) { alert('Chyba CLIP: ' + d.error); return; }
    currentPhotoScore = d.photo_score;
    const pct = d.photo_score_pct;
    document.getElementById('photo-score-fill').style.width = pct + '%';
    document.getElementById('photo-score-val').textContent = pct + ' / 100';
    document.getElementById('photo-score-row').style.display = 'flex';
  } catch(e) {
    console.warn('Chyba uploadu fotky:', e);
  }
}

async function go() {
  const adr=document.getElementById('adr').value.trim();
  const pl=document.getElementById('pl').value;
  const di=document.getElementById('di').value;
  if(!adr||!pl){alert('Vyplň adresu a plochu.');return;}
  const btn=document.querySelector('.btn');
  btn.disabled=true; btn.textContent='Klasifikuji…';
  document.getElementById('geo').textContent='';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({adresa:adr,plocha:parseFloat(pl),dispozice:parseInt(di),
        poi_override: poiUserModified ? poi : {},
        photo_score:currentPhotoScore})});
    const d=await r.json();
    if(d.error){alert(d.error);return;}
    document.getElementById('geo').textContent='📍 '+d.resolved;
    if(d.poi) setSliders(d.poi);
    const tier=d.tier, cfg=T[tier];
    document.getElementById('badge').textContent=cfg.l;
    document.getElementById('badge').className='badge '+cfg.c;
    document.getElementById('conf').textContent='Jistota modelu: '+d.confidence+' %';
    const src=d.median_source==='city'?`mediánem v lokalitě ${d.city_used}`:'celostátním mediánem dispozice';
    const msgs={0:`Byt se nachází <strong>pod ${src}</strong> (<strong>${fmt(d.ref_median)}</strong>). Je levnější než většina srovnatelných bytů.`,
      1:`Byt odpovídá <strong>${src}</strong> (<strong>${fmt(d.ref_median)}</strong>). Je v průměru srovnatelných bytů.`,
      2:`Byt se nachází <strong>nad ${src}</strong> (<strong>${fmt(d.ref_median)}</strong>). Je dražší než většina srovnatelných bytů.`};
    const ctx=document.getElementById('ctx');
    ctx.className='ctx ctx'+tier;
    ctx.innerHTML=msgs[tier]+`<div class="note">Hranice: levné do <strong>${fmt(d.t1_abs)}</strong> | střední <strong>${fmt(d.t1_abs)}–${fmt(d.t2_abs)}</strong> | drahé nad <strong>${fmt(d.t2_abs)}</strong></div>`;
    document.getElementById('mk').style.left=[17,50,83][tier]+'%';
    document.getElementById('dname').textContent=d.disp_name;
    document.getElementById('rl1').textContent=fmt(d.t1_abs);
    document.getElementById('rl2').textContent=fmt(d.t2_abs);
    document.getElementById('rlmax').textContent=fmt(d.t2_abs*1.6)+'+';
    document.getElementById('zones').innerHTML=T.map((t,i)=>`<div class="zone z${i}${tier===i?' active':''}"><div class="zt">${i===0?'do '+fmt(d.t1_abs):i===1?fmt(d.t1_abs)+'–'+fmt(d.t2_abs):'nad '+fmt(d.t2_abs)}</div><div class="zb">${t.l}</div></div>`).join('');
    const p=d.probabilities;
    document.getElementById('cbars').innerHTML=T.map((t,i)=>`<div class="cbar"><span class="clbl">${t.l}</span><div class="cbg"><div class="cfill" style="width:${Math.round(p[i]*100)}%;background:${t.col}"></div></div><span class="cval">${Math.round(p[i]*100)}%</span></div>`).join('');
    const pr=d.poi;
    const photoRow=d.photo_score!==null?[['Kvalita fotek (CLIP)',Math.round(d.photo_score*100)+' / 100']]:[];
    document.getElementById('dg').innerHTML=[
      ['MHD zastávka',pr.mhd_m+' m'],['Škola',pr.skola_m+' m'],['Park',pr.park_m+' m'],
      ['Lékárna',pr.lekarna_m+' m'],['Supermarket',pr.supermarket_m+' m'],
      ['Median lokality',fmt(d.ref_median)],...photoRow,
      ['GPS',d.lat.toFixed(4)+', '+d.lon.toFixed(4)]
    ].map(([l,v])=>`<div class="di"><div class="dl">${l}</div><div class="dv">${v}</div></div>`).join('');
    const res=document.getElementById('result');
    res.style.display='block';res.style.animation='none';res.offsetHeight;res.style.animation='fi .4s ease';
  }catch(e){alert('Chyba: '+e.message);}
  finally{btn.disabled=false;btn.textContent='Zařadit do cenového pásma';}
}
</script>
</body></html>"""


@app.route("/")
def index():
    return render_template_string(HTML, n_data=5740,
                                  has_photo=has_photo,
                                  clip_available=CLIP_AVAILABLE)


@app.route("/score_photo", methods=["POST"])
def score_photo_endpoint():
    if not CLIP_AVAILABLE:
        return jsonify({"error": "CLIP model není dostupný."})
    if "photo" not in request.files:
        return jsonify({"error": "Žádná fotka."})
    img_bytes = request.files["photo"].read()
    try:
        score = score_photo(img_bytes)
        return jsonify({"photo_score": score,
                        "photo_score_pct": round(score * 100)})
    except Exception as e:
        return jsonify({"error": f"Chyba zpracování fotky: {e}"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    adresa = data["adresa"]
    plocha = float(data["plocha"])
    disp = int(data["dispozice"])
    poi_override = data.get("poi_override", {})
    client_photo_score = data.get("photo_score", 0.5)

    lat, lon, resolved = geocode(adresa)
    if lat is None:
        return jsonify({"error": f"Adresu '{adresa}' se nepodařilo najít."})

    poi = get_poi_from_gps(lat, lon)

    key_map = {"mhd": "mhd_m", "skola": "skola_m",
               "park": "park_m", "lekarna": "lekarna_m", "super": "supermarket_m"}
    for k, col in key_map.items():
        if k in poi_override:
            poi[col] = float(poi_override[k])

    city = find_city(resolved)
    mhd_d = min(1.0, round(poi["mhd_m"] / 1000, 3))
    ref_median, src = get_ref_median(city, disp)
    cast_enc = encode_mesto(city or "neznámo")

    fv_dict = {
        "plocha_m2": plocha,
        "dispozice_skore": disp,
        "lat": lat,
        "lon": lon,
        "mhd_dostupnost": mhd_d,
        "skola_m": poi["skola_m"],
        "park_m": poi["park_m"],
        "mhd_m": poi["mhd_m"],
        "lekarna_m": poi["lekarna_m"],
        "supermarket_m": poi["supermarket_m"],
        "photo_score": client_photo_score,
    }

    fv_num = np.array([[fv_dict[f] for f in input_features]], dtype=np.float32)
    fv_num = scaler.transform(fv_num)
    fv_cat = np.array([[cast_enc]], dtype=int)

    proba = model.predict([fv_num, fv_cat], verbose=0)[0]
    tier = int(np.argmax(proba))
    conf = round(float(np.max(proba)) * 100, 1)

    t1_abs = round(ref_median * t1_rel)
    t2_abs = round(ref_median * t2_rel)

    disp_names = {1: "Garsoniéra", 2: "1+kk / 1+1", 3: "2+kk / 2+1",
                  4: "3+kk / 3+1", 5: "4+kk a více"}

    return jsonify({
        "tier": tier,
        "confidence": conf,
        "probabilities": [round(float(p), 3) for p in proba],
        "ref_median": round(ref_median),
        "t1_abs": t1_abs,
        "t2_abs": t2_abs,
        "median_source": "city" if src != "ČR" else "national",
        "city_used": src,
        "disp_name": disp_names.get(disp, f"Dispozice {disp}"),
        "photo_score": fv_dict["photo_score"] if has_photo else None,
        "lat": lat,
        "lon": lon,
        "resolved": resolved,
        "poi": {k: round(v) for k, v in poi.items()},
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)