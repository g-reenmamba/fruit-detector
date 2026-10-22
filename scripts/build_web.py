"""
build_web.py
------------
Builds the always-on browser version of Fruit Detector into docs/ (served by
GitHub Pages). Everything runs in the visitor's browser, so there is no server
to fall asleep.

    python3 scripts/build_web.py

It writes:
  docs/model/clip_visual_w8.onnx   CLIP image encoder, ONNX, weight-only int8 (~89 MB)
  docs/model/prompts.bin / .json   pre-encoded fresh / rotten / not-produce prompts
  docs/fonts, docs/samples, docs/img/background.png   copied from the app

Why weight-only int8: ONNX Runtime's standard dynamic quantization also
quantizes activations and dropped the 71-photo test from 69 to 65-66 correct.
Storing only the weights as int8 (one scale per output channel) and
de-quantizing them on load keeps 69/71 at a quarter of the fp32 size.

Needs: torch, open_clip_torch, onnx (pip install onnx onnxruntime).
The browser runtime (onnxruntime-web 1.29.0, MIT) is vendored in docs/vendor/ort/,
copied unmodified from the npm package. The page itself is docs/index.html,
docs/style.css and docs/app.js.
"""

import json
import os
import shutil
import sys

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import clip_detector as cd  # noqa: E402

DOCS = os.path.join(ROOT, "docs")


def export_onnx(path):
    det = cd.load_detector()
    torch.onnx.export(det["visual"].eval(), (torch.randn(2, 3, 224, 224),), path,
                      input_names=["pixels"], output_names=["embedding"],
                      dynamic_axes={"pixels": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return det


def weight_only_int8(src, dst):
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    uses = {}
    for node in g.node:
        for inp in node.input:
            if inp in inits:
                uses.setdefault(inp, []).append(node)
    new_inits, new_nodes = [], []
    for name, init in list(inits.items()):
        arr = numpy_helper.to_array(init)
        users = uses.get(name, [])
        if arr.size < 100_000 or arr.dtype != np.float32 or not users:
            continue
        if any(n.op_type not in ("MatMul", "Gemm", "Conv") for n in users):
            continue
        node = users[0]
        if node.op_type == "MatMul":
            axis = arr.ndim - 1                                   # [K, N] -> per output column
        elif node.op_type == "Gemm":
            axis = 0 if next((a.i for a in node.attribute if a.name == "transB"), 0) else 1
        else:
            axis = 0                                              # Conv [O, I, kh, kw]
        reduce = tuple(i for i in range(arr.ndim) if i != axis)
        scale = (np.abs(arr).max(axis=reduce) / 127.0).astype(np.float32)
        scale[scale == 0] = 1e-8
        shape = [1] * arr.ndim
        shape[axis] = -1
        q = np.clip(np.round(arr / scale.reshape(shape)), -127, 127).astype(np.int8)
        g.initializer.remove(init)
        new_inits += [numpy_helper.from_array(q, name + "_q"), numpy_helper.from_array(scale, name + "_scale"),
                      numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + "_zp")]
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale", name + "_zp"],
                                          [name], axis=axis, name=name + "_dq"))
    g.initializer.extend(new_inits)
    for n in reversed(new_nodes):
        g.node.insert(0, n)
    onnx.checker.check_model(m)
    onnx.save(m, dst)


def main():
    os.makedirs(os.path.join(DOCS, "model"), exist_ok=True)
    tmp = os.path.join(DOCS, "model", "_fp32.onnx")
    det = export_onnx(tmp)
    weight_only_int8(tmp, os.path.join(DOCS, "model", "clip_visual_w8.onnx"))
    os.remove(tmp)

    vecs = torch.cat([det["fresh"], det["rotten"], det["not_produce"]]).numpy().astype("<f4")
    n = len(det["produce"])
    vecs.tofile(os.path.join(DOCS, "model", "prompts.bin"))
    with open(os.path.join(DOCS, "model", "prompts.json"), "w") as fh:
        json.dump({"produce": det["produce"], "not_produce": det["not_produce_prompts"],
                   "dim": int(vecs.shape[1]), "logit_scale": det["logit_scale"],
                   "unsure_below": cd.UNSURE_BELOW, "not_produce_above": cd.NOT_PRODUCE_ABOVE,
                   # CLIP colour statistics, same as clip_detector.PREPROCESS
                   "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711],
                   # [start, end) row ranges inside prompts.bin
                   "rows": {"fresh": [0, n], "rotten": [n, 2 * n], "not_produce": [2 * n, len(vecs)]}},
                  fh, indent=1)

    for sub in ("samples", os.path.join("static", "fonts")):
        dst = os.path.join(DOCS, os.path.basename(sub))
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(os.path.join(ROOT, sub), dst)
    os.makedirs(os.path.join(DOCS, "img"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "assets", "background.png"), os.path.join(DOCS, "img", "background.png"))
    shutil.copy(os.path.join(ROOT, "src", "produce_info.json"), os.path.join(DOCS, "model", "produce_info.json"))
    open(os.path.join(DOCS, ".nojekyll"), "w").close()   # serve files as-is, no Jekyll
    size = os.path.getsize(os.path.join(DOCS, "model", "clip_visual_w8.onnx")) / 1e6
    print(f"built docs/ (model {size:.1f} MB, {len(det['produce'])} produce types)")


if __name__ == "__main__":
    main()
