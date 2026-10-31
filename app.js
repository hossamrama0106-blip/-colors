import { env, CLIPVisionModelWithProjection, Tensor } from './vendor/transformers.min.js';
env.backends.onnx.wasm.wasmPaths = new URL('vendor/', location.href).href;
env.backends.onnx.wasm.numThreads = 1;
env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = new URL('models/', location.href).href;
const $ = s => document.querySelector(s);
const MODEL = 'Xenova/clip-vit-base-patch32';
const MEAN = [0.48145466, 0.4578275, 0.40821073], STD = [0.26862954, 0.26130258, 0.27577711];

let model, catalog = {}, items = [], E, dim, calib, mu;

function f16(u) { // half-float -> float
  const s = (u & 0x8000) >> 15, e = (u & 0x7c00) >> 10, f = u & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
async function init() {
  const [cat, idx, bin, cal] = await Promise.all([
    fetch('data/catalog.json').then(r => r.json()),
    fetch('data/index.json').then(r => r.json()),
    fetch('data/emb.bin').then(r => r.arrayBuffer()),
    fetch('data/calib.json').then(r => r.json()).catch(() => null)
  ]);
  cat.forEach(p => catalog[p.id] = p);
  items = idx.items; dim = idx.dim; calib = cal || { exact: 0.9, close: 0.7, weak: 0.5 };
  const u = new Uint16Array(bin); E = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) E[i] = f16(u[i]);
  // mean-centre all embeddings: spreads similarities so unrelated items score ~0
  const N = u.length / dim; mu = new Float32Array(dim);
  for (let i = 0; i < N; i++) for (let j = 0; j < dim; j++) mu[j] += E[i * dim + j] / N;
  for (let i = 0; i < N; i++) {
    let n = 0; for (let j = 0; j < dim; j++) { E[i * dim + j] -= mu[j]; n += E[i * dim + j] ** 2; }
    n = Math.sqrt(n) || 1; for (let j = 0; j < dim; j++) E[i * dim + j] /= n;
  }
  model = await CLIPVisionModelWithProjection.from_pretrained(MODEL);
  setStatus('جاهز ✔ — ' + Object.keys(catalog).length + ' منتج على الموقع، ' + items.length + ' صورة مرجعية');
}
function setStatus(t) { $('#status').textContent = t; }

function toTensor(img) {
  const c = document.createElement('canvas'); c.width = c.height = 224;
  const x = c.getContext('2d'); x.fillStyle = '#fff'; x.fillRect(0, 0, 224, 224);
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height, s = 224 / Math.max(w, h);
  const nw = Math.max(1, Math.round(w * s)), nh = Math.max(1, Math.round(h * s));
  x.imageSmoothingQuality = 'high'; x.drawImage(img, (224 - nw) >> 1, (224 - nh) >> 1, nw, nh);
  const d = x.getImageData(0, 0, 224, 224).data, out = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) for (let k = 0; k < 3; k++) out[k * 224 * 224 + i] = (d[i * 4 + k] / 255 - MEAN[k]) / STD[k];
  return new Tensor('float32', out, [1, 3, 224, 224]);
}
async function embed(img) {
  const { image_embeds } = await model({ pixel_values: toTensor(img) });
  const v = Float32Array.from(image_embeds.data); let n = 0; for (const a of v) n += a * a; n = Math.sqrt(n);
  const c = v.map((a, j) => a / n - mu[j]); let m = 0; for (const a of c) m += a * a; m = Math.sqrt(m) || 1;
  return c.map(a => a / m);
}
function search(q) {
  const N = items.length, sims = new Float32Array(N);
  for (let i = 0; i < N; i++) { let s = 0; const o = i * dim; for (let j = 0; j < dim; j++) s += q[j] * E[o + j]; sims[i] = s; }
  const prod = new Map(), loc = new Map();
  for (let i = 0; i < N; i++) {
    const it = items[i], s = sims[i];
    if (it.t === 'c') { const o = prod.get(it.p); if (!o || s > o.s) prod.set(it.p, { id: it.p, s }); }
    else if (it.m) { // archive photo of a product that is still on the site -> evidence for that product
      const v = Math.min(s, it.ms), o = prod.get(it.m); if (!o || v > o.s) prod.set(it.m, { id: it.m, s: v });
    } else { const o = loc.get(it.l); if (!o || s > o.s) loc.set(it.l, { label: it.l, s, th: it.th }); }
  }
  const P = [...prod.values()].filter(r => catalog[r.id]).sort((a, b) => b.s - a.s), L = [...loc.values()].sort((a, b) => b.s - a.s);
  return { P, L };
}
// cosine -> display percentage (piecewise linear through calibration anchors)
function pct(s) {
  const a = [[0.3, 0], [calib.weak, 30], [calib.close, 60], [calib.exact, 95], [1, 100]];
  if (s <= a[0][0]) return 0;
  for (let i = 1; i < a.length; i++) if (s <= a[i][0]) { const [x0, y0] = a[i - 1], [x1, y1] = a[i]; return Math.round(y0 + (y1 - y0) * (s - x0) / (x1 - x0)); }
  return 100;
}
function level(s) {
  if (s >= calib.exact) return ['b-ok', 'تطابق شبه مؤكد'];
  if (s >= calib.close) return ['b-mid', 'مقارب'];
  if (s >= calib.weak) return ['b-bad', 'تشابه ضعيف — غير مؤكد'];
  return ['b-bad', 'لم يتم التأكد'];
}
const color = s => s >= calib.exact ? 'var(--ok)' : s >= calib.close ? 'var(--mid)' : 'var(--bad)';
const esc = t => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const cleanLabel = l => l.split('/').slice(-2).join(' / ');
const catImg = p => p.images[0] + (p.images[0].includes('?') ? '&' : '?') + 'width=400';

function prodCard(r, best) {
  const p = catalog[r.id], lv = level(r.s), pc = pct(r.s);
  const stock = p.available ? '<span class="badge b-av">متاح</span>' : '<span class="badge b-old">نفذ من الموقع</span>';
  return '<div class="card ' + (best ? 'best' : '') + '"><img src="' + catImg(p) + '" loading="lazy" alt="">' +
    '<div class="info"><h3>' + esc(p.title) + '</h3>' + stock + '<span class="badge ' + lv[0] + '">' + lv[1] + '</span>' +
    '<div class="bar"><i style="width:' + pc + '%;background:' + color(r.s) + '"></i></div>' +
    '<div class="small">نسبة التطابق ' + pc + '% (تشابه ' + (r.s * 100).toFixed(1) + ')' + (p.price ? ' • السعر ' + p.price + ' ج.م' : '') + '</div>' +
    '<a href="' + p.url + '" target="_blank" rel="noopener">فتح المنتج في الموقع ↗</a></div></div>';
}
function oldCard(r, best) {
  const lv = level(r.s), pc = pct(r.s);
  return '<div class="card ' + (best ? 'best' : '') + '"><img src="' + r.th + '" loading="lazy" alt="">' +
    '<div class="info"><h3>' + esc(cleanLabel(r.label)) + '</h3><span class="badge b-old">منتج قديم نفذ</span><span class="badge ' + lv[0] + '">' + lv[1] + '</span>' +
    '<div class="bar"><i style="width:' + pc + '%;background:' + color(r.s) + '"></i></div>' +
    '<div class="small">نسبة التطابق ' + pc + '% — من أرشيف الموديلات (غير متاح على الموقع حاليًا)</div></div></div>';
}
function render(src, { P, L }) {
  const bp = P[0], bl = L[0];
  let h = '<div class="queryRow"><img src="' + src + '" alt="">';
  const oldWins = bl && bl.s > bp.s + 0.02 && bl.s >= calib.close;
  if (bp.s < calib.weak && (!bl || bl.s < calib.weak)) {
    h += '<div class="verdict" style="color:var(--bad)">❓ لم يتم التأكد — لا يوجد منتج مشابه بدرجة كافية. أقرب النتائج:</div>';
  } else if (oldWins) {
    h += '<div class="verdict">🕘 غالبًا هذا <b>منتج قديم نفذ</b> (غير متاح على الموقع). أقرب منتج متاح موضح أسفله.</div>';
  } else if (bp.s >= calib.exact) {
    h += '<div class="verdict" style="color:var(--ok)">✅ هذا المنتج هو: <b>' + esc(catalog[bp.id].title) + '</b></div>';
  } else if (bp.s >= calib.close) {
    h += '<div class="verdict" style="color:var(--mid)">≈ مقارب لـ <b>' + esc(catalog[bp.id].title) + '</b> (غير مؤكد 100%)</div>';
  } else {
    h += '<div class="verdict" style="color:var(--bad)">⚠ تشابه ضعيف — أقرب موديلات:</div>';
  }
  if (oldWins) h += '<div class="group"><h2>الأقرب في الأرشيف (قديم)</h2>' + L.slice(0, 2).map((r, i) => oldCard(r, i === 0)).join('') + '</div>';
  h += '<div class="group"><h2>' + (oldWins ? 'أقرب منتجات متاحة على الموقع' : 'أقرب منتجات الموقع') + '</h2>' +
    P.slice(0, oldWins ? 3 : 5).map((r, i) => prodCard(r, i === 0 && !oldWins)).join('') + '</div>';
  if (!oldWins && L.length && L[0].s > calib.weak) h += '<div class="group"><h2>موديلات قديمة مشابهة (نفذت)</h2>' + L.slice(0, 3).map(r => oldCard(r)).join('') + '</div>';
  $('#results').insertAdjacentHTML('afterbegin', h + '</div>');
}
async function handle(file) {
  if (!model) { setStatus('النموذج لم يُحمَّل بعد، انتظر لحظات…'); return; }
  const url = URL.createObjectURL(file); let img;
  try { img = await createImageBitmap(file); } catch (e) { setStatus('تعذّر قراءة الصورة'); return; }
  $('#preview').src = url; $('#preview').hidden = false;
  setStatus('جاري التحليل…');
  const q = await embed(img); render(url, search(q)); setStatus('تم ✔');
}
const files = fs => [...fs].filter(f => f && f.type.startsWith('image/')).forEach(handle);
document.addEventListener('paste', e => files([...e.clipboardData.items].map(i => i.getAsFile()).filter(Boolean)));
$('#pasteBtn').addEventListener('click', async () => {
  try {
    const found = [];
    for (const it of await navigator.clipboard.read()) {
      const t = it.types.find(x => x.startsWith('image/'));
      if (t) found.push(new File([await it.getType(t)], 'paste.' + t.split('/')[1], { type: t }));
    }
    if (found.length) files(found); else setStatus('لا توجد صورة في الحافظة — انسخ صورة أولًا');
  } catch (e) { setStatus('تعذّر الوصول للحافظة — اضغط Ctrl+V أو (لصق) بالضغط المطوّل على الصفحة'); }
});
const drop = $('#drop');
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => files(e.dataTransfer.files));
$('#file').addEventListener('change', e => { files(e.target.files); e.target.value = ''; });
$('#cam').addEventListener('change', e => { files(e.target.files); e.target.value = ''; });
init().catch(e => { console.error(e); setStatus('خطأ في التحميل: ' + e.message); });
