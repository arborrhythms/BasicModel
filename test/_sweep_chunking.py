"""Sweep MM_20M PerceptualSpace <chunking> options to compare reconstruction.

Builds temp configs from MM_20M.xml with a WORKING 1024-wide InputSpace
(the 5-wide narrowing crashes the event mux), varying only <chunking>.
Runs each end-to-end and reports the reconstruction lines. Not a pytest test.
"""
import os, re, subprocess, sys, time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
EPOCHS = int(os.environ.get("SWEEP_EPOCHS", "200"))
CHUNKS = os.environ.get("SWEEP_CHUNKS", "analyse,none,radix,lexicon,bpe,mphf").split(",")

with open(os.path.join(PROJECT, "data", "MM_20M.xml")) as f:
    base = f.read()

# Restore a WORKING 1024-wide InputSpace + PS handoff (the 5-wide IS crashes).
_lexer = os.environ.get("SWEEP_LEXER", "raw")
base = re.sub(r"  <InputSpace>.*?</InputSpace>",
              "  <InputSpace>\n    <nOutput>8192</nOutput>\n"
              "    <nVectors>256</nVectors>\n    <nDim>1024</nDim>\n"
              f"    <lexer>{_lexer}</lexer>\n  </InputSpace>",
              base, count=1, flags=re.DOTALL)
base = base.replace("<nInputDim>5</nInputDim>", "<nInputDim>1024</nInputDim>")
_order = os.environ.get("SWEEP_ORDER")
if _order:
    base = re.sub(r"<conceptualOrder>\d+</conceptualOrder>",
                  f"<conceptualOrder>{_order}</conceptualOrder>", base)
_promo = os.environ.get("SWEEP_PROMO")
if _promo:
    base = re.sub(r"(\s*)(<chunking>\w+</chunking>)",
                  rf"\1<chunkPromotionThreshold>{_promo}</chunkPromotionThreshold>\1\2",
                  base, count=1)
_recon = os.environ.get("SWEEP_RECON")
if _recon is not None:
    base = re.sub(r"<reconstructionScale>[\d.]+</reconstructionScale>",
                  f"<reconstructionScale>{_recon}</reconstructionScale>", base)
base = re.sub(r"<numEpochs>\d+</numEpochs>", f"<numEpochs>{EPOCHS}</numEpochs>", base)
_lr = os.environ.get("SWEEP_LR")
if _lr:
    base = re.sub(r"<learningRate>[\d.]+</learningRate>", f"<learningRate>{_lr}</learningRate>", base)
_cb = os.environ.get("SWEEP_CODEBOOK")
if _cb:
    # PS codebook is commented out in MM_20M (-> default quantize); SS = quantize.
    base = base.replace("<!-- codebook>quantize</codebook -->", f"<codebook>{_cb}</codebook>")
    base = base.replace("<codebook>quantize</codebook>", f"<codebook>{_cb}</codebook>")
# Independent PS/SS codebook overrides (PS is commented in MM_20M; SS = quantize).
_ps_cb = os.environ.get("SWEEP_PS_CB")
if _ps_cb:
    base = base.replace("<!-- codebook>quantize</codebook -->", f"<codebook>{_ps_cb}</codebook>")
_ss_cb = os.environ.get("SWEEP_SS_CB")
if _ss_cb:
    base = base.replace("<codebook>quantize</codebook>", f"<codebook>{_ss_cb}</codebook>")
# SS commitmentBeta / l1Lambda overrides (insert into the SymbolicSpace block).
_ins = ""
_ssc = os.environ.get("SWEEP_SS_COMMIT")
if _ssc is not None:
    _ins += f"    <commitmentBeta>{_ssc}</commitmentBeta>\n"
_ssl = os.environ.get("SWEEP_SS_L1")
if _ssl is not None:
    _ins += f"    <l1Lambda>{_ssl}</l1Lambda>\n"
if _ins:
    base = base.replace("  </SymbolicSpace>", _ins + "  </SymbolicSpace>")

env = dict(os.environ, MODEL_COMPILE="none")
for chunk in CHUNKS:
    cfg = re.sub(r"<chunking>\w+</chunking>", f"<chunking>{chunk}</chunking>", base)
    path = os.path.join(PROJECT, "data", f"_sweep_{chunk}.xml")
    with open(path, "w") as f:
        f.write(cfg)
    t0 = time.time()
    try:
        r = subprocess.run([PY, "bin/Models.py", f"data/_sweep_{chunk}.xml"],
                           cwd=PROJECT, env=env, capture_output=True, text=True, timeout=600)
        out = r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired:
        out = "TIMEOUT"
    finally:
        os.remove(path)
    dt = time.time() - t0
    recon = re.findall(r"row\[\d+\] input=.*?(?:OK|MISMATCH)", out)
    errs = re.findall(r"(?:RuntimeError|ValueError|invalid for input|inconsisten).*", out)
    print(f"\n===== chunking={chunk}  ({dt:.0f}s, epochs={EPOCHS}) =====", flush=True)
    if errs:
        print("  CRASH/ERR:", errs[-1][:160])
        if os.environ.get("SWEEP_TRACE"):
            for l in out.strip().splitlines()[-22:]:
                print("    | " + l[:150])
    for line in re.findall(r"=(?:BPE|RDX|REV)=.*", out)[-12:]:
        print("  " + line[:175])
    if recon:
        for line in recon[:6]:
            print("  " + line[:130])
    elif not errs:
        print("  (no reconstruction lines found)")
