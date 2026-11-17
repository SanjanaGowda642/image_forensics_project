"""Flask web app:  upload an image (or use the built-in demo) and get
original | tampering overlay + arrows | binary mask | artifact cut-out | restored image | scores.

    python run.py          # opens http://127.0.0.1:5000
"""
import base64
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, send_from_directory

from data.dataset_generator import make_demo_forgery
from utils.pipeline import ForensicsPipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
_pipe = None


def pipeline():
    global _pipe
    if _pipe is None:
        _pipe = ForensicsPipeline()
    return _pipe


def _b64(img, ext=".png"):
    ok, enc = cv2.imencode(ext, img)
    return "data:image/png;base64," + base64.b64encode(enc.tobytes()).decode()


def _pick():
    """Which region counts as the original (0 = automatic choice, 1.. = alternatives)."""
    try:
        return max(0, int(request.values.get("pick", 0)))
    except ValueError:
        return 0


def _respond(img, name, pick=0):
    name = f"{name}_{int(time.time())}"
    res = pipeline().analyze(img, OUT_DIR, name, pick=pick)
    body = {k: _b64(res[k]) for k in ("original", "overlay", "mask", "cutout", "restored", "panel")}
    body["info"] = res["info"]
    body["saved_as"] = name
    return jsonify(body)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="no file uploaded"), 400
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="could not decode image"), 400
    return _respond(img, os.path.splitext(os.path.basename(f.filename or "upload"))[0], _pick())


@app.route("/api/demo")
def demo():
    forged, _ = make_demo_forgery(seed=int(request.args.get("seed", 7)))
    return _respond(forged, "demo", _pick())


@app.route("/api/status")
def status():
    p = pipeline()
    return jsonify(cnn_loaded=p.net is not None, cnn_error=p.cnn_error)


@app.route("/outputs/<path:fn>")
def outputs(fn):
    return send_from_directory(OUT_DIR, fn)


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Tampering Detection &amp; Restoration</title>
<style>
:root{--bg:#e8ecef;--ink:#14222e;--mut:#5b6b78;--navy:#17346e;--orange:#f58c1e;--card:#fff;--line:#c9d2d9;
--red:#d6453a;--green:#3f9d5a;--amber:#e9a02a}
@media (prefers-color-scheme:dark){:root{--bg:#10171d;--ink:#e6edf2;--mut:#93a4b1;--card:#18232c;--line:#2b3a46;--navy:#7da2ee}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
header{padding:28px clamp(16px,4vw,48px) 8px}
h1{margin:0;font:700 clamp(24px,3.4vw,38px)/1.15 Georgia,"Times New Roman",serif;letter-spacing:-.01em}
header p{margin:6px 0 0;color:var(--mut);max-width:70ch}
main{padding:16px clamp(16px,4vw,48px) 60px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:14px 0 22px}
button,.btn{font:600 14px "Segoe UI",system-ui,sans-serif;border:2px solid var(--navy);background:var(--navy);color:#fff;
padding:10px 18px;border-radius:6px;cursor:pointer}
button.ghost,.btn.ghost{background:transparent;color:var(--navy)}
button:focus-visible,input:focus-visible+label{outline:3px solid var(--orange);outline-offset:2px}
button:disabled{opacity:.5;cursor:wait}
input[type=file]{position:absolute;opacity:0;width:1px;height:1px}
#drop{border:2px dashed var(--line);border-radius:8px;padding:12px 16px;color:var(--mut);flex:1;min-width:220px}
#drop.over{border-color:var(--orange);color:var(--ink)}
#status{color:var(--mut);min-height:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
figure img{display:block;width:100%;height:auto;background:#000}
figcaption{display:flex;justify-content:space-between;align-items:center;padding:9px 12px;font-weight:600;font-size:13.5px}
figcaption a{font-weight:600;color:var(--navy);text-decoration:none;border-bottom:1px solid currentColor;font-size:13px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;margin-top:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px}
.card h3{margin:0 0 6px;font-size:14px;color:var(--mut);font-weight:600}
.gauge svg{width:100%;max-width:240px;display:block;margin:auto}
.label{font:700 24px Georgia,serif}
.label.bad{color:var(--red)}.label.ok{color:var(--green)}.label.warn{color:#c77700}
#warn{display:none;margin:0 0 18px;border-left:5px solid var(--orange);font-weight:600}
#swapbar{display:none;margin:0 0 18px;align-items:center;gap:14px;flex-wrap:wrap}
#swapbar p{margin:0;flex:1;min-width:240px;color:var(--mut);font-size:14px}
dl{margin:10px 0 0;display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:14px}
dt{color:var(--mut)}dd{margin:0;font-weight:600}
#res{display:none}
.panel{margin-top:22px}
.panel img{width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
@media (prefers-reduced-motion:no-preference){.spin{animation:s 1s linear infinite;display:inline-block}@keyframes s{to{transform:rotate(360deg)}}}
</style></head><body>
<header><h1>Image tampering detection &amp; restoration</h1>
<p>Finds copy-move forgeries, marks every cloned region with its source, isolates the noise artifact, and rebuilds the area underneath.</p></header>
<main>
<div class="bar">
  <input id="file" type="file" accept="image/*">
  <label class="btn" for="file">Choose image</label>
  <button class="ghost" id="demo">Run built-in demo</button>
  <div id="drop">…or drop an image here</div>
</div>
<div id="status" role="status"></div>
<div id="warn" class="card" role="alert"></div>
<div id="res">
  <div id="swapbar" class="card">
    <p id="swaptxt"></p>
    <button id="swap" class="ghost">Wrong object marked? Try another source</button>
  </div>
  <div class="grid">
    <figure><img id="i_original" alt="Original"><figcaption>1 · Original image<a id="d_original" download>save</a></figcaption></figure>
    <figure><img id="i_overlay" alt="Overlay"><figcaption>2 · Tampering overlay + source→target arrows<a id="d_overlay" download>save</a></figcaption></figure>
    <figure><img id="i_mask" alt="Binary mask"><figcaption>3 · Binary tampering mask<a id="d_mask" download>save</a></figcaption></figure>
    <figure><img id="i_cutout" alt="Artifact cutout"><figcaption>4 · Artifact cut-out (zoom-in)<a id="d_cutout" download>save</a></figcaption></figure>
    <figure><img id="i_restored" alt="Restored"><figcaption>5 · Inpainted / restored result<a id="d_restored" download>save</a></figcaption></figure>
  </div>
  <div class="metrics">
    <div class="card gauge"><h3>Confidence score</h3><div id="g1"></div></div>
    <div class="card gauge"><h3>Tampering intensity</h3><div id="g2"></div></div>
    <div class="card"><h3>Predicted label</h3><div id="lab" class="label"></div><dl id="det"></dl></div>
  </div>
  <div class="panel"><div class="card"><h3>Report panel <a id="d_panel" download style="float:right;color:var(--navy)">save</a></h3><img id="i_panel" alt="Report panel"></div></div>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
function gauge(el,v){
  const cx=120,cy=115,r=88,seg=['#3f9d5a','#e9a02a','#d6453a'];
  const pt=(a,rad)=>[cx+rad*Math.cos(a),cy+rad*Math.sin(a)];
  let s='<svg viewBox="0 0 240 160" role="img" aria-label="'+v.toFixed(2)+'">';
  for(let i=0;i<3;i++){const a0=Math.PI+i*Math.PI/3,a1=a0+Math.PI/3,p=pt(a0,r),q=pt(a1,r);
    s+=`<path d="M${p[0]} ${p[1]} A${r} ${r} 0 0 1 ${q[0]} ${q[1]}" stroke="${seg[i]}" stroke-width="16" fill="none"/>`}
  const a=Math.PI+Math.PI*Math.min(Math.max(v,0),1),n=pt(a,r-8);
  s+=`<line x1="${cx}" y1="${cy}" x2="${n[0]}" y2="${n[1]}" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><circle cx="${cx}" cy="${cy}" r="7" fill="currentColor"/>`;
  s+=`<text x="${cx}" y="${cy+30}" text-anchor="middle" font-size="26" font-weight="700" fill="currentColor">${v.toFixed(2)}</text>`;
  s+=`<text x="${cx}" y="${cy+47}" text-anchor="middle" font-size="13" fill="currentColor" opacity=".7">(${Math.round(v*100)}%)</text></svg>`;
  el.innerHTML=s}
function show(d,keepScroll){
  for(const k of ['original','overlay','mask','cutout','restored','panel']){$('i_'+k).src=d[k];const a=$('d_'+k);a.href=d[k];a.download=d.saved_as+'_'+k+'.png'}
  const i=d.info;gauge($('g1'),i.confidence);gauge($('g2'),i.intensity);
  const L=$('lab');L.textContent=i.label;L.className='label '+(i.label==='AUTHENTIC'?'ok':i.label==='INCONCLUSIVE'?'warn':'bad');
  const W=$('warn');W.textContent=i.warning||'';W.style.display=i.warning?'block':'none';
  $('det').innerHTML=[['Confidence',i.confidence.toFixed(2)+' ('+Math.round(i.confidence*100)+'%)'],
   ['Intensity',i.intensity.toFixed(2)+' ('+Math.round(i.intensity*100)+'%)'],['Manipulated area',i.area_pct+'%'],
   ['Source regions matched',i.regions_matched],['Image resolution',i.resolution],['Keypoint inliers',i.inliers],
   ['CNN probability',i.cnn_probability==null?'n/a (no weights)':i.cnn_probability]].concat(i.cropped?[['Analysed picture region',i.crop_box[2]+'×'+i.crop_box[3]+' at ('+i.crop_box[0]+', '+i.crop_box[1]+')']]:[]).concat([['Time',i.seconds+' s']])
   .map(x=>'<dt>'+x[0]+'</dt><dd>'+x[1]+'</dd>').join('');
  const sb=$('swapbar'),multi=i.n_regions>=2&&i.label==='COPY-MOVE FORGERY';
  sb.style.display=multi?'flex':'none';
  if(multi)$('swaptxt').textContent=(i.ambiguous
    ?'Amber regions A, B… are identical copies and the image gives no clue which is the original; the restored picture removes the assumed copy. '
    :'Red = suspected copy (removed in the restored image), yellow = original. Automatic choice can be wrong when the clone blends in well. ')+
    'Showing option '+(i.pick+1)+' of '+i.n_regions+'.';
  $('res').style.display='block';if(!keepScroll)$('res').scrollIntoView({behavior:'smooth'})}
let last=null, pick=0;
async function run(p, keepScroll){
  $('status').innerHTML='<span class="spin">◐</span> Analysing…';$('demo').disabled=true;$('swap').disabled=true;
  try{const r=await p;const d=await r.json();if(d.error)throw new Error(d.error);show(d,keepScroll);$('status').textContent='Done — files saved in the outputs/ folder.'}
  catch(e){$('status').textContent='Error: '+e.message}
  $('demo').disabled=false;$('swap').disabled=false}
function send(keepScroll){
  if(last.file){const fd=new FormData();fd.append('image',last.file);fd.append('pick',pick);
    run(fetch('/api/analyze',{method:'POST',body:fd}),keepScroll)}
  else run(fetch('/api/demo?pick='+pick),keepScroll)}
function upload(f){last={file:f};pick=0;send(false)}
$('file').onchange=e=>e.target.files[0]&&upload(e.target.files[0]);
$('demo').onclick=()=>{last={demo:true};pick=0;send(false)};
$('swap').onclick=()=>{pick+=1;send(true)};
const dz=$('drop');
['dragenter','dragover'].forEach(t=>dz.addEventListener(t,e=>{e.preventDefault();dz.classList.add('over')}));
['dragleave','drop'].forEach(t=>dz.addEventListener(t,e=>{e.preventDefault();dz.classList.remove('over')}));
dz.addEventListener('drop',e=>e.dataTransfer.files[0]&&upload(e.dataTransfer.files[0]));
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
