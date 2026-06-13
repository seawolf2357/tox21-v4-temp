#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초전도 후보 ML 스크리닝 — H200 GPU 8장 풀가동
CHGNet(범용 ML 원자간 퍼텐셜)으로 AlB2족 붕화물/탄화물 ~990 후보를
GPU 8장에 분산해 구조 안정성(relax + 에너지) 대량 스크리닝.
  - 역할: 수백~수천 후보 → 안정 수십개로 거르기 (Tc 단정 아님)
  - 안정 + 가벼운원소(높은 포논) 후보 → QE 전자-포논 정밀(별도, H100)
정직: ML은 '합성가능 안정구조 거르기'. 진짜 Tc는 QE가 판정.
"""
import os, sys, json, time, itertools

HOME = os.path.expanduser("~")
WD = os.path.join(HOME, "vidraft", "sc_screen")
os.makedirs(WD, exist_ok=True)
RESULT = os.path.join(WD, "result.json")
LOG = os.path.join(WD, "run.log")

def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(s, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(s + "\n")
    except Exception:
        pass

def ensure_pkgs():
    try:
        import pymatgen  # noqa
        import chgnet    # noqa
        return
    except ImportError:
        log("installing chgnet + pymatgen (pip --user) ...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q",
                        "chgnet", "pymatgen"], check=False)

# ---------- 후보 원소 ----------
METALS = ['Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
          'Ni','Cu','Zn','Ga','Sr','Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd',
          'In','Sn','Ba','La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Tl','Pb']
# 가벼운 원소 원자량 (높은 Debye 온도 → 초전도 잠재 보너스)
MASS = {'Li':6.9,'Be':9.0,'B':10.8,'C':12.0,'N':14.0,'Na':23.0,'Mg':24.3,'Al':27.0}

def make(spec, a=3.05, c=3.30):
    from pymatgen.core import Structure, Lattice
    if len(spec) == 3:                      # 단층 AlB2-type: M, X, X
        M, X1, X2 = spec
        lat = Lattice.hexagonal(a, c)
        return Structure(lat, [M, X1, X2],
                         [[0,0,0],[1/3,2/3,0.5],[2/3,1/3,0.5]])
    else:                                   # 2층 교대 금속 도핑: M1 / M2
        M1, M2 = spec
        lat = Lattice.hexagonal(a, 2*c)
        return Structure(lat, [M1,'B','B',M2,'B','B'],
                         [[0,0,0.0],[1/3,2/3,0.25],[2/3,1/3,0.25],
                          [0,0,0.5],[1/3,2/3,0.75],[2/3,1/3,0.75]])

def gen():
    out = []
    for M in METALS:
        out.append(("%sB2" % M, (M,'B','B')))      # 단일 붕화물 MB2
    for M in METALS:
        out.append(("%sC2" % M, (M,'C','C')))      # 탄화물 MC2
        out.append(("%sBeB" % M, (M,'Be','B')))    # 혼합 X-site
    for M1, M2 in itertools.combinations(METALS, 2):
        out.append(("%s%sB4" % (M1, M2), (M1, M2)))  # 이원 금속 도핑
    return out

# ---------- GPU worker (1 GPU = 1 프로세스) ----------
def worker(rank, chunk, q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import warnings; warnings.filterwarnings("ignore")
    try:
        from chgnet.model import CHGNet, StructOptimizer
        model = CHGNet.load(verbose=False)
        opt = StructOptimizer(model)
    except Exception as ex:
        q.put((rank, [{"name": "_FATAL_", "error": str(ex)[:200]}])); return
    res = []
    for name, spec in chunk:
        try:
            st = make(spec)
            r = opt.relax(st, steps=40, fmax=0.10, verbose=False)
            fs = r["final_structure"]
            e = float(model.predict_structure(fs)["e"])     # eV/atom (corrected)
            res.append({"name": name, "e_per_atom": round(e, 4),
                        "natoms": len(fs),
                        "vol_pa": round(fs.volume/len(fs), 3),
                        "a": round(fs.lattice.a, 3), "c": round(fs.lattice.c, 3)})
        except Exception as ex:
            res.append({"name": name, "error": str(ex)[:60]})
        if len(res) % 25 == 0:
            log("GPU%d progress %d/%d" % (rank, len(res), len(chunk)))
    q.put((rank, res))

def light_score(name):
    import re
    el = re.findall(r"[A-Z][a-z]?", name)
    el = [e for e in el if e in MASS or len(e) <= 2]
    if not el:
        return 0.0
    inv = [1.0/MASS.get(e, 65.0) for e in el]    # 무거운 원소=65 페널티
    return round(sum(inv)/len(inv), 4)

def main():
    ensure_pkgs()
    try:
        import torch
        vis = torch.cuda.device_count()
    except Exception:
        vis = "?"
    cands = gen()
    log("candidates=%d  torch_visible_gpus=%s" % (len(cands), vis))
    import torch.multiprocessing as tmp
    ctx = tmp.get_context("spawn")
    q = ctx.Queue()
    NG = 8                                   # H200 8장 강제 분산
    chunks = [cands[i::NG] for i in range(NG)]
    procs = []
    for r in range(NG):
        p = ctx.Process(target=worker, args=(r, chunks[r], q))
        p.start(); procs.append(p)
    allres = []
    for _ in range(NG):
        rank, out = q.get()
        allres.extend(out)
        log("collected GPU%d (%d)" % (rank, len(out)))
    for p in procs:
        p.join()
    ok = [r for r in allres if "e_per_atom" in r]
    for r in ok:
        r["light"] = light_score(r["name"])
        # 점수: 안정(에너지↓) + 가벼운원소(초전도 잠재↑). 낮을수록 우수.
        r["score"] = round(r["e_per_atom"] - 6.0*r["light"], 4)
    ok.sort(key=lambda x: x["score"])
    json.dump({"n_total": len(allres), "n_stable": len(ok),
               "note": "ML 1차 거르기 — Tc 단정 아님. 상위 후보는 QE 전자-포논 정밀 필요.",
               "ranked": ok[:60]}, open(RESULT, "w"), indent=2)
    log("DONE stable=%d/%d  RESULT=%s" % (len(ok), len(allres), RESULT))
    log("TOP20: " + ", ".join("%s(%.3f)" % (r["name"], r["score"]) for r in ok[:20]))

if __name__ == "__main__":
    main()
