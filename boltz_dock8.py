#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boltz-2 신약 도킹 — H200 GPU 8장 100% 풀가동
OGA(O-GlcNAcase, 알츠하이머/타우병증 타겟, UniProt O60502)에 약물후보 도킹.
1단계(이 스크립트): 8개 테스트분자를 GPU 8장에 각 1개씩 병렬 도킹 → 작동+GPU100% 확인.
성공하면 2단계로 진짜 OGA 신약후보 대량 도킹.
Boltz-2 = 무거운 3D 구조예측 신경망 → GPU 확실히 가득.
"""
import os, sys, subprocess, time, json, glob, urllib.request

HOME = os.path.expanduser("~")
WD = os.path.join(HOME, "vidraft", "boltz_dock")
os.makedirs(WD, exist_ok=True)
LOG = os.path.join(WD, "dock.log")

def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(s, flush=True)
    try:
        with open(LOG, "a") as f: f.write(s + "\n")
    except Exception: pass

def ensure_boltz():
    try:
        import boltz  # noqa
        log("boltz already installed")
        return
    except ImportError:
        log("installing boltz (수 GB 가중치는 첫 predict시 자동 다운로드)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q", "boltz"], check=False)

def get_oga_seq():
    # OGA catalytic domain (UniProt O60502). 너무 길면 무겁지만 GPU 풀가동엔 좋음.
    try:
        f = urllib.request.urlopen("https://rest.uniprot.org/uniprotkb/O60502.fasta", timeout=40).read().decode()
        seq = "".join(f.strip().split("\n")[1:])
        log("OGA seq from UniProt: %d aa" % len(seq))
        # 너무 길면(916) Boltz 메모리 과다 → catalytic domain 60~400 정도로 절단(테스트)
        return seq[:380] if len(seq) > 380 else seq
    except Exception as e:
        log("UniProt 실패(%s) → fallback 짧은 도메인" % str(e)[:40])
        return ("MAQKGSGAPLDPAALAALAAPLDPRALAAALAALAAPLDPAALAALAAPLAALAAPLDPAALAA")

# 1단계 테스트 분자 8개 (검증된 drug-like — GPU 작동 확인용; 2단계서 진짜 OGA후보로 교체)
LIGANDS = [
    ("aspirin",       "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine",      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("naproxen",      "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("metformin",     "CN(C)C(=N)NC(=N)N"),
    ("glucose",       "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("salicylate",    "OC(=O)c1ccccc1O"),
]

def dock_worker(rank, seq, name, smiles):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    wd = os.path.join(WD, "lig%d_%s" % (rank, name))
    os.makedirs(wd, exist_ok=True)
    yml = os.path.join(wd, "in.yaml")
    with open(yml, "w") as f:
        f.write("version: 1\n")
        f.write("sequences:\n")
        f.write("  - protein:\n      id: A\n      sequence: \"%s\"\n" % seq)
        f.write("  - ligand:\n      id: B\n      smiles: \"%s\"\n" % smiles)
        f.write("properties:\n  - affinity:\n      binder: B\n")
    t0 = time.time()
    try:
        r = subprocess.run(
            ["boltz", "predict", yml, "--use_msa_server", "--out_dir", os.path.join(wd, "out"),
             "--diffusion_samples", "1", "--override"],
            capture_output=True, text=True, timeout=7200)
        ok = r.returncode == 0
        aff = None
        for jf in glob.glob(os.path.join(wd, "out", "**", "affinity*.json"), recursive=True):
            try:
                aff = json.load(open(jf)); break
            except Exception: pass
        log("GPU%d %s: rc=%d %.0fs affinity=%s" % (rank, name, r.returncode, time.time()-t0, aff))
        if not ok:
            log("GPU%d %s STDERR: %s" % (rank, name, (r.stderr or "")[-300:]))
    except subprocess.TimeoutExpired:
        log("GPU%d %s TIMEOUT(2h)" % (rank, name))
    except Exception as e:
        log("GPU%d %s ERROR: %s" % (rank, name, str(e)[:200]))

def main():
    ensure_boltz()
    seq = get_oga_seq()
    import torch.multiprocessing as tmp
    ctx = tmp.get_context("spawn")
    procs = []
    for r in range(8):
        name, smi = LIGANDS[r]
        p = ctx.Process(target=dock_worker, args=(r, seq, name, smi))
        p.start(); procs.append(p)
        log("launched GPU%d -> %s" % (r, name))
    for p in procs:
        p.join()
    log("ALL DONE — 8 GPU 도킹 완료. 결과는 %s/lig*/out/" % WD)

if __name__ == "__main__":
    main()
