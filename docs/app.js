/*
 * Fruit Detector, static edition: CLIP's image encoder runs in the browser with
 * ONNX Runtime Web, so the page needs no server and never goes to sleep.
 * Same maths as src/clip_detector.py in the Python app.
 */
(() => {
  "use strict";
  const MODEL_URL = "model/clip_visual_w8.onnx";
  const CACHE_NAME = "fruit-detector-model-v1";
  const SAMPLES = ["fresh_apple.png", "fresh_cherry.png", "fresh_grapes.png", "fresh_mango.png",
    "fresh_pineapple.png", "fresh_raspberry.png", "fresh_watermelon.png", "rotten_apple.png",
    "rotten_pear.png", "rotten_pear_2.png", "rotten_strawberry.png", "rotten_tomato.png"];

  // ---------- pixel sprites (ASCII art -> crisp SVG) ----------
  function sprite(rows, palette, px, label) {
    const h = rows.length, w = Math.max(...rows.map(r => r.length));
    let rects = "";
    rows.forEach((row, y) => [...row].forEach((ch, x) => {
      if (palette[ch]) rects += `<rect x="${x}" y="${y}" width="1" height="1" fill="${palette[ch]}"/>`;
    }));
    const a11y = label ? `role="img" aria-label="${label}"` : 'aria-hidden="true"';
    return `<svg ${a11y} width="${w * px}" height="${h * px}" viewBox="0 0 ${w} ${h}" shape-rendering="crispEdges" xmlns="http://www.w3.org/2000/svg">${rects}</svg>`;
  }
  const APPLE_ROWS = ["......s.....", ".....s.LL...", "..RRRsRRLL..", ".RhhRRRRRR..", "RhhRRRRRRRR.",
    "RhRRRRRRRRRd", "RRRRRRRRRRRd", "RRRRRRRRRRdd", ".RRRRRRRRRd.", ".RRRRRRRRdd.", "..RRdRRddd..", "...dd..dd..."];
  const APPLE_PAL = { R: "#E2546B", h: "#FFB3BF", d: "#A8324F", s: "#6B4A3A", L: "#6CC08B" };
  const ICON = {
    fresh: sprite(["........GG", ".......GGG", "......GGG.", "GG...GGG..", "GGG.GGG...", ".GGGGG....", "..GGG....."], { G: "#3E8A5A" }, 4),
    rotten: sprite([".WWWWW.", "WWWWWWW", "WKKWKKW", "WKKWKKW", "WWWKWWW", ".WWWWW.", ".W.W.W."], { W: "#FFFFFF", K: "#6E3A63" }, 4),
    unsure: sprite([".QQQQ.", "QQ..QQ", "....QQ", "...QQ.", "..QQ..", "......", "..QQ.."], { Q: "#9A7724" }, 4),
  };
  document.getElementById("apple").innerHTML = sprite(APPLE_ROWS, APPLE_PAL, 5, "pixel apple");

  // ---------- elements & state ----------
  const $ = id => document.getElementById(id);
  const fileInput = $("file"), dropzone = $("dropzone"), select = $("sample"), photo = $("photo"),
    checkBtn = $("check"), verdictEl = $("verdict"), fileHint = $("file-hint");
  let session = null, prompts = null, currentImage = null, busy = false, loadError = null;

  const cells = (p, kind) => `<div class="cells ${kind}" role="img" aria-label="${kind} ${Math.floor(p * 100)}%">` +
    Array.from({ length: 10 }, (_, i) => `<div class="cell${i < Math.round(p * 10) ? " on" : ""}"></div>`).join("") + "</div>";
  const pct = p => (p >= 0.995 ? "over 99%" : `${Math.floor(p * 100)}%`);

  function showWaiting(text) {
    verdictEl.innerHTML = `<div class="waiting"><span class="ghost">${ICON.unsure}</span><p>${text}</p></div>`;
  }
  function showLoading(done, total) {
    const p = total ? done / total : 0;
    verdictEl.innerHTML = `<div class="waiting"><p>Loading the AI model into your browser.<br>
      First visit only: ${Math.round(done / 1e6)} of ${Math.round((total || 89e6) / 1e6)} MB.</p>
      <div style="width:100%;max-width:300px">${cells(p, "load")}</div></div>`;
  }
  function showEmptyPhoto() {
    photo.className = "photo empty";
    photo.innerHTML = `<span class="ghost">${sprite(APPLE_ROWS, APPLE_PAL, 5)}</span><p>Your fruit shows up here.</p>`;
  }
  function refreshButton() {
    checkBtn.disabled = !currentImage || !session || busy;
    checkBtn.textContent = busy ? "Looking closely..." : (!session && !loadError ? "Loading model..." : "Check freshness");
  }
  function idleMessage() {
    if (loadError) showWaiting(loadError);
    else if (!session) return;  // loading progress is on screen
    else showWaiting("Upload a photo or pick a sample, then press Check freshness.");
  }

  // ---------- model loading (cached after the first visit) ----------
  async function fetchModel() {
    let cache = null;
    try { cache = await caches.open(CACHE_NAME); } catch (e) { cache = null; }
    if (cache) {
      const hit = await cache.match(MODEL_URL);
      if (hit) return hit.arrayBuffer();
    }
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`model download failed (${res.status})`);
    const total = Number(res.headers.get("content-length")) || 89e6;
    const reader = res.body.getReader(), chunks = [];
    let done = 0, last = 0;
    for (;;) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      chunks.push(value); done += value.length;
      if (done - last > 1.5e6) { last = done; showLoading(done, total); }
    }
    const buf = new Uint8Array(done);
    let off = 0; for (const c of chunks) { buf.set(c, off); off += c.length; }
    if (cache) { try { await cache.put(MODEL_URL, new Response(buf)); } catch (e) { /* quota: fine */ } }
    return buf.buffer;
  }

  async function loadModel() {
    showLoading(0, 89e6);
    try {
      const [meta, bin] = await Promise.all([
        fetch("model/prompts.json").then(r => r.json()),
        fetch("model/prompts.bin").then(r => r.arrayBuffer()),
      ]);
      prompts = { ...meta, vecs: new Float32Array(bin) };
      // absolute URL: the runtime loads its .mjs with import(), which rejects bare relative paths
      ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      const model = await fetchModel();
      verdictEl.innerHTML = `<div class="waiting"><p>Almost ready, setting up the model...</p></div>`;
      session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    } catch (err) {
      console.error(err);
      loadError = "The model couldn't load in this browser. Try a recent Chrome, Edge, Firefox or Safari.";
    }
    refreshButton();
    idleMessage();
  }

  // ---------- preprocessing: same as CLIP (short side 224, centre crop, CLIP colour stats) ----------
  function downscaled(img) {
    // halve repeatedly first so big photos are filtered smoothly, not aliased
    let src = img, w = img.naturalWidth, h = img.naturalHeight;
    while (Math.min(w, h) > 448) {
      const c = document.createElement("canvas");
      c.width = Math.round(w / 2); c.height = Math.round(h / 2);
      const ctx = c.getContext("2d"); ctx.imageSmoothingQuality = "high";
      ctx.drawImage(src, 0, 0, c.width, c.height);
      src = c; w = c.width; h = c.height;
    }
    return { src, w, h };
  }
  function toTensor(img) {
    const { src, w, h } = downscaled(img);
    const side = Math.min(w, h), sx = (w - side) / 2, sy = (h - side) / 2;
    const data = new Float32Array(2 * 3 * 224 * 224);
    [false, true].forEach((mirror, b) => {
      const c = document.createElement("canvas"); c.width = c.height = 224;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.imageSmoothingQuality = "high";
      if (mirror) { ctx.translate(224, 0); ctx.scale(-1, 1); }
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 224, 224);  // transparent PNGs -> white, like PIL convert("RGB") on most
      ctx.drawImage(src, sx, sy, side, side, 0, 0, 224, 224);
      const px = ctx.getImageData(0, 0, 224, 224).data;
      for (let i = 0; i < 224 * 224; i++) {
        for (let ch = 0; ch < 3; ch++) {
          data[b * 3 * 224 * 224 + ch * 224 * 224 + i] = (px[i * 4 + ch] / 255 - prompts.mean[ch]) / prompts.std[ch];
        }
      }
    });
    return new ort.Tensor("float32", data, [2, 3, 224, 224]);
  }

  // ---------- scoring: softmax over "fresh X" / "rotten X" captions ----------
  function analyse(emb) {
    const d = prompts.dim, f = new Float32Array(d);
    for (let b = 0; b < 2; b++) {
      let n = 0; for (let k = 0; k < d; k++) n += emb[b * d + k] ** 2;
      n = Math.sqrt(n); for (let k = 0; k < d; k++) f[k] += emb[b * d + k] / n;
    }
    let n = 0; for (let k = 0; k < d; k++) n += f[k] ** 2; n = Math.sqrt(n);
    for (let k = 0; k < d; k++) f[k] /= n;
    const rows = prompts.rows.not_produce[1], logits = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      let s = 0; for (let k = 0; k < d; k++) s += prompts.vecs[r * d + k] * f[k];
      logits[r] = prompts.logit_scale * s;
    }
    const softmax = (a, b) => {
      let m = -Infinity; for (let i = a; i < b; i++) m = Math.max(m, logits[i]);
      const e = []; let z = 0; for (let i = a; i < b; i++) { const v = Math.exp(logits[i] - m); e.push(v); z += v; }
      return e.map(v => v / z);
    };
    const np = prompts.produce.length, p = softmax(0, 2 * np);
    let rotten = 0; for (let i = np; i < 2 * np; i++) rotten += p[i];
    let best = 0, bestP = -1;
    for (let i = 0; i < np; i++) { const v = p[i] + p[np + i]; if (v > bestP) { bestP = v; best = i; } }
    const all = softmax(0, rows);
    let notProduce = 0; for (let i = 2 * np; i < rows; i++) notProduce += all[i];
    return { rotten, fresh: 1 - rotten, produce: prompts.produce[best], produceConf: bestP, notProduce };
  }

  function render(r) {
    if (r.notProduce > prompts.not_produce_above) {
      verdictEl.innerHTML = `<div class="verdict not_produce">${ICON.unsure}<h3>That's not a fruit!</h3>
        <p>This doesn't look like a fruit or vegetable. Try a photo of one piece of produce.</p></div>`;
      return;
    }
    const pred = r.rotten > 0.5 ? "rotten" : "fresh", conf = Math.max(r.rotten, r.fresh);
    const label = conf >= prompts.unsure_below ? pred : "unsure";
    const head = {
      fresh: ["Looks fresh!", `Good to eat, ${pct(conf)} sure.`],
      rotten: ["Looks rotten.", `Better not eat it, ${pct(conf)} sure.`],
      unsure: ["Hmm, not sure.", `It leans ${pred}, but it's only ${pct(conf)} sure. A closer, well-lit photo of one fruit helps.`],
    }[label];
    const val = p => (p < 0.01 ? "<1" : Math.floor(p * 100)) + "%";
    verdictEl.innerHTML = `<div class="verdict ${label}">${ICON[label]}<h3>${head[0]}</h3><p>${head[1]}</p></div>
      <div class="scores">
        <span class="n">Fresh</span>${cells(r.fresh, "fresh")}<span class="v">${val(r.fresh)}</span>
        <span class="n">Rotten</span>${cells(r.rotten, "rotten")}<span class="v">${val(r.rotten)}</span>
      </div>` + (r.produceConf >= 0.3 ? `<p class="guess">Looks like: <b>${r.produce}</b></p>` : "");
  }

  async function check() {
    if (!currentImage || !session) return;
    busy = true; refreshButton();
    await new Promise(requestAnimationFrame);  // let the button repaint first
    try {
      const out = await session.run({ pixels: toTensor(currentImage) });
      render(analyse(out.embedding.data));
    } catch (err) {
      console.error(err);
      showWaiting("Something went wrong reading that photo. Try another one.");
    }
    busy = false; refreshButton();
  }

  // ---------- choosing a photo ----------
  function setImage(url, name) {
    const img = new Image();
    img.onload = () => {
      currentImage = img;
      photo.className = "photo"; photo.innerHTML = "";
      img.className = "photo-img"; img.alt = name ? `Photo: ${name}` : "Your photo";
      photo.appendChild(img);
      refreshButton(); idleMessage();
    };
    img.onerror = () => { showWaiting("That file isn't an image this browser can open. Try a JPG or PNG."); };
    img.src = url;
  }
  function useFile(file) {
    if (!file) return;
    if (!/^image\//.test(file.type)) { showWaiting("That file isn't an image. Try a JPG, PNG or WEBP photo."); return; }
    select.value = "";
    fileHint.textContent = file.name;
    setImage(URL.createObjectURL(file), file.name);
  }
  const sampleName = f => {
    const [state, ...rest] = f.replace(/\.png$/, "").split("_");
    const another = rest[rest.length - 1] === "2";
    const fruit = rest.filter(x => x !== "2").join(" ");
    return `${state[0].toUpperCase()}${state.slice(1)} ${fruit}${another ? " (another one)" : ""}`;
  };
  SAMPLES.forEach(f => select.add(new Option(sampleName(f), f)));
  select.addEventListener("change", () => {
    if (!select.value) return;
    fileInput.value = ""; fileHint.textContent = "or drop a photo here. JPG, PNG or WEBP.";
    setImage(`samples/${select.value}`, sampleName(select.value));
  });
  fileInput.addEventListener("change", () => useFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", e => useFile(e.dataTransfer.files[0]));
  checkBtn.addEventListener("click", check);

  showEmptyPhoto();
  refreshButton();
  loadModel();
})();
