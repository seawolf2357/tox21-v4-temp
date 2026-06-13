#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초전도 후보 ML 스크리닝 v2 — H200 GPU 8장
v1 문제 수정:
  1) Formation energy (CHGNet element reference) → 조성 무관 진짜 안정성
  2) 안정성 게이트 (formation > 0.15 eV/atom 제거) — v1은 987/987 다 통과
  3) 초전도 친화 점수 = 경원소비율 + 저질량(높은 Debye/포논)
검증: MgB2(39K) · BeB2 가 상위권에 드는지로 점수 신뢰성 확인.
정직: ML은 'MgB2-like 경량 안정 금속화합물' 선별. 진짜 Tc는 QE 전자-포논.
"""
import os, sys, json, time, itertools

HOME = os.path.expanduser("~")
WD = os.path.join(HOME, "vidraft", "sc_screen")
os.makedirs(WD, exist_ok=True)
RESULT = os.path.join(WD, "result_v2.json")
REFS_PATH = os.path.join(WD, "elem_refs.json")
LOG = os.path.join(WD, "run_v2.log")

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
        import pymatgen, chgnet, ase  # noqa
        return
    except ImportError:
        log("installing chgnet pymatgen ...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q",
                        "chgnet", "pymatgen"], check=False)

METALS = ['Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
          'Ni','Cu','Zn','Ga','Sr','Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd',
          'In','Sn','Ba','La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Tl','Pb']
LIGHT = {'H','Li','Be','B','C','N'}      # 경원소(높은 포논 주파수)

def make(spec, a=3.05, c=3.30):
    from pymatgen.core import Structure, Lattice
    if len(spec) == 3:
        M, X1, X2 = spec
        lat = Lattice.hexagonal(a, c)
        return Structure(lat, [M, X1, X2],
                         [[0,0,0],[1/3,2/3,0.5],[2/3,1/3,0.5]])
    M1, M2 = spec
    lat = Lattice.hexagonal(a, 2*c)
    return Structure(lat, [M1,'B','B',M2,'B','B'],
                     [[0,0,0.0],[1/3,2/3,0.25],[2/3,1/3,0.25],
                      [0,0,0.5],[1/3,2/3,0.75],[2/3,1/3,0.75]])

def gen():
    out = []
    for M in METALS:
        out.append(("%sB2" % M, (M,'B','B')))
    for M in METALS:
        out.append(("%sC2" % M, (M,'C','C')))
        out.append(("%sBeB" % M, (M,'Be','B')))
    for M1, M2 in itertools.combinations(METALS, 2):
        out.append(("%s%sB4" % (M1, M2), (M1, M2)))
    return out

# ---------- element reference (GPU0, 1회) ----------
def refs_worker(elements, path, q):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import warnings; warnings.filterwarnings("ignore")
    from chgnet.model import CHGNet, StructOptimizer
    from ase.build import bulk
    from pymatgen.io.ase import AseAtomsAdaptor
    model = CHGNet.load(verbose=False); opt = StructOptimizer(model)
    refs = {}
    for el in elements:
        at = None
        for args in ((el,), (el,'fcc',4.0,True), (el,'bcc',3.3,True), (el,'sc',3.0,True)):
            try:
                at = bulk(*args) if len(args) == 1 else bulk(args[0], args[1], a=args[2], cubic=args[3])
                break
            except Exception:
                continue
        if at is None:
            refs[el] = 0.0; continue
        try:
            st = AseAtomsAdaptor.get_structure(at)
            r = opt.relax(st, steps=80, fmax=0.05, verbose=False)
            refs[el] = float(model.predict_structure(r["final_structure"])["e"])
        except Exception:
            refs[el] = 0.0
    json.dump(refs, open(path, "w"))
    q.put(len([k for k,v in refs.items() if v != 0.0]))

# ---------- candidate worker (8 GPU) ----------
def cand_worker(rank, chunk, refs, q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import warnings; warnings.filterwarnings("ignore")
    from chgnet.model import CHGNet, StructOptimizer
    from pymatgen.core import Element
    model = CHGNet.load(verbose=False); opt = StructOptimizer(model)
    out = []
    for name, spec in chunk:
        try:
            st = make(spec)
            r = opt.relax(st, steps=40, fmax=0.10, verbose=False)
            fs = r["final_structure"]
            e = float(model.predict_structure(fs)["e"])
            frac = fs.composition.fractional_composition.get_el_amt_dict()
            ref_e = sum(f*refs.get(el, 0.0) for el, f in frac.items())
            formation = e - ref_e
            lf = sum(f for el, f in frac.items() if el in LIGHT)
            am = sum(f*Element(el).atomic_mass for el, f in frac.items())
            out.append({"name": name, "formation": round(formation,4),
                        "light_frac": round(lf,3), "avg_mass": round(am,2),
                        "vol_pa": round(fs.volume/len(fs),3)})
        except Exception as ex:
            out.append({"name": name, "error": str(ex)[:60]})
        if len(out) % 30 == 0:
            log("GPU%d %d/%d" % (rank, len(out), len(chunk)))
    q.put((rank, out))

def sc_score(r):
    # 안정 후보 중: 경원소비율↑ + 저질량↑ 일수록 우수(낮음). MgB2-like.
    return round(-r["light_frac"] - 30.0/max(r["avg_mass"], 1.0), 4)

def main():
    ensure_pkgs()
    elements = sorted(set(METALS + ['B','C','Be']))
    cands = gen()
    log("cands=%d  refs=%d elements" % (len(cands), len(elements)))
    import torch.multiprocessing as tmp
    ctx = tmp.get_context("spawn")
    # 1) element references
    rq = ctx.Queue()
    rp = ctx.Process(target=refs_worker, args=(elements, REFS_PATH, rq))
    rp.start(); nref = rq.get(); rp.join()
    refs = json.load(open(REFS_PATH))
    log("element refs done (%d non-zero)" % nref)
    # 2) candidates on 8 GPUs
    NG = 8
    chunks = [cands[i::NG] for i in range(NG)]
    q = ctx.Queue(); procs = []
    for r in range(NG):
        p = ctx.Process(target=cand_worker, args=(r, chunks[r], refs, q))
        p.start(); procs.append(p)
    allres = []
    for _ in range(NG):
        rank, out = q.get(); allres.extend(out); log("collected GPU%d" % rank)
    for p in procs:
        p.join()
    done = [r for r in allres if "formation" in r]
    stable = [r for r in done if r["formation"] < 0.15]      # 안정성 게이트
    for r in stable:
        r["score"] = sc_score(r)
    stable.sort(key=lambda x: x["score"])
    # MgB2 / BeB2 검증
    def chk(nm):
        hit = [r for r in stable if r["name"] == nm]
        if not hit:
            d = [r for r in done if r["name"] == nm]
            return "%s: GATED-OUT (formation=%s)" % (nm, d[0]["formation"] if d else "NA")
        rk = stable.index(hit[0]) + 1
        return "%s: rank %d/%d (top %d%%) formation=%.3f score=%.3f" % (
            nm, rk, len(stable), 100*rk//len(stable), hit[0]["formation"], hit[0]["score"])
    log("VERIFY " + chk("MgB2"))
    log("VERIFY " + chk("BeB2"))
    json.dump({"n_done": len(done), "n_stable": len(stable),
               "verify": {"MgB2": chk("MgB2"), "BeB2": chk("BeB2")},
               "note": "score=경원소+저질량 (MgB2-like). 안정게이트 formation<0.15. Tc는 QE 필요.",
               "ranked": stable[:60]}, open(RESULT, "w"), indent=2)
    log("DONE stable=%d/%d" % (len(stable), len(done)))
    log("TOP20: " + ", ".join("%s(%.2f)" % (r["name"], r["score"]) for r in stable[:20]))

if __name__ == "__main__":
    main()
