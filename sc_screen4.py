#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초전도 후보 ML 스크리닝 v4 — H200 GPU 8장 초대규모 (~124,000 후보)
v3(MgB2 top2% 검증된 점수) + 사원계 M1M2M3M4B8 (C(42,4)=111,930) 추가.
relax step 20으로 경량화(후보 많음). GPU 8장 수시간 풀가동.
정직: 점수는 MgB2-like 경량 안정 붕화물 선별. Tc 판정은 QE.
"""
import os, sys, json, time, itertools

HOME = os.path.expanduser("~")
WD = os.path.join(HOME, "vidraft", "sc_screen")
os.makedirs(WD, exist_ok=True)
RESULT = os.path.join(WD, "result_v4.json")
REFS_PATH = os.path.join(WD, "elem_refs.json")
LOG = os.path.join(WD, "run_v4.log")

def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(s, flush=True)
    try:
        with open(LOG, "a") as f: f.write(s + "\n")
    except Exception: pass

def ensure_pkgs():
    try:
        import pymatgen, chgnet, ase  # noqa
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q",
                        "chgnet", "pymatgen"], check=False)

METALS = ['Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
          'Ni','Cu','Zn','Ga','Sr','Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd',
          'In','Sn','Ba','La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Tl','Pb']
LIGHT = {'H','Li','Be','B','C','N'}

def make(spec, a=3.05, c=3.30):
    from pymatgen.core import Structure, Lattice
    tag = spec[0]
    if tag == 'mono':
        M, X1, X2 = spec[1:]
        lat = Lattice.hexagonal(a, c)
        return Structure(lat, [M, X1, X2],
                         [[0,0,0],[1/3,2/3,0.5],[2/3,1/3,0.5]])
    if tag == 'bi':
        M1, M2 = spec[1:]
        lat = Lattice.hexagonal(a, 2*c)
        return Structure(lat, [M1,'B','B',M2,'B','B'],
                         [[0,0,0.0],[1/3,2/3,0.25],[2/3,1/3,0.25],
                          [0,0,0.5],[1/3,2/3,0.75],[2/3,1/3,0.75]])
    if tag == 'tri':
        M1, M2, M3 = spec[1:]
        lat = Lattice.hexagonal(a, 3*c)
        return Structure(lat, [M1,'B','B',M2,'B','B',M3,'B','B'],
                         [[0,0,0.0],[1/3,2/3,1/6],[2/3,1/3,1/6],
                          [0,0,1/3],[1/3,2/3,0.5],[2/3,1/3,0.5],
                          [0,0,2/3],[1/3,2/3,5/6],[2/3,1/3,5/6]])
    # quad: M1/M2/M3/M4 (4층)
    M1, M2, M3, M4 = spec[1:]
    lat = Lattice.hexagonal(a, 4*c)
    return Structure(lat, [M1,'B','B',M2,'B','B',M3,'B','B',M4,'B','B'],
                     [[0,0,0.0],[1/3,2/3,1/8],[2/3,1/3,1/8],
                      [0,0,1/4],[1/3,2/3,3/8],[2/3,1/3,3/8],
                      [0,0,1/2],[1/3,2/3,5/8],[2/3,1/3,5/8],
                      [0,0,3/4],[1/3,2/3,7/8],[2/3,1/3,7/8]])

def gen():
    out = []
    for M in METALS:
        out.append(("%sB2" % M, ('mono', M, 'B', 'B')))
    for M in METALS:
        out.append(("%sC2" % M, ('mono', M, 'C', 'C')))
        out.append(("%sBeB" % M, ('mono', M, 'Be', 'B')))
    for M1, M2 in itertools.combinations(METALS, 2):
        out.append(("%s%sB4" % (M1, M2), ('bi', M1, M2)))
    for M1, M2, M3 in itertools.combinations(METALS, 3):
        out.append(("%s%s%sB6" % (M1, M2, M3), ('tri', M1, M2, M3)))
    for M1, M2, M3, M4 in itertools.combinations(METALS, 4):
        out.append(("%s%s%s%sB8" % (M1, M2, M3, M4), ('quad', M1, M2, M3, M4)))
    return out

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
    q.put(len([k for k, v in refs.items() if v != 0.0]))

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
            r = opt.relax(st, steps=20, fmax=0.12, verbose=False)
            fs = r["final_structure"]
            e = float(model.predict_structure(fs)["e"])
            frac = fs.composition.fractional_composition.get_el_amt_dict()
            ref_e = sum(f*refs.get(el, 0.0) for el, f in frac.items())
            formation = e - ref_e
            lf = sum(f for el, f in frac.items() if el in LIGHT)
            am = sum(f*Element(el).atomic_mass for el, f in frac.items())
            out.append({"name": name, "formation": round(formation,4),
                        "light_frac": round(lf,3), "avg_mass": round(am,2)})
        except Exception:
            pass
        if len(out) % 1000 == 0 and len(out) > 0:
            log("GPU%d %d/%d" % (rank, len(out), len(chunk)))
    q.put((rank, out))

def sc_score(r):
    return round(-r["light_frac"] - 30.0/max(r["avg_mass"], 1.0), 4)

def main():
    ensure_pkgs()
    elements = sorted(set(METALS + ['B','C','Be']))
    cands = gen()
    log("cands=%d  refs=%d" % (len(cands), len(elements)))
    import torch.multiprocessing as tmp
    ctx = tmp.get_context("spawn")
    rq = ctx.Queue()
    rp = ctx.Process(target=refs_worker, args=(elements, REFS_PATH, rq))
    rp.start(); nref = rq.get(); rp.join()
    refs = json.load(open(REFS_PATH))
    log("element refs done (%d)" % nref)
    NG = 8
    chunks = [cands[i::NG] for i in range(NG)]
    q = ctx.Queue(); procs = []
    for r in range(NG):
        p = ctx.Process(target=cand_worker, args=(r, chunks[r], refs, q))
        p.start(); procs.append(p)
    allres = []
    for _ in range(NG):
        rank, out = q.get(); allres.extend(out); log("collected GPU%d (%d)" % (rank, len(out)))
    for p in procs: p.join()
    done = allres
    stable = [r for r in done if r["formation"] < 0.15]
    for r in stable:
        r["score"] = sc_score(r)
    stable.sort(key=lambda x: x["score"])
    def chk(nm):
        hit = [r for r in stable if r["name"] == nm]
        if not hit: return "%s: absent" % nm
        rk = stable.index(hit[0]) + 1
        return "%s: rank %d/%d (top %d%%)" % (nm, rk, len(stable), 100*rk//max(len(stable),1))
    log("VERIFY " + chk("MgB2") + " | " + chk("BeB2") + " | " + chk("LiB2"))
    json.dump({"n_done": len(done), "n_stable": len(stable),
               "verify": {"MgB2": chk("MgB2"), "BeB2": chk("BeB2"), "LiB2": chk("LiB2")},
               "ranked": stable[:100]}, open(RESULT, "w"), indent=2)
    log("DONE stable=%d/%d" % (len(stable), len(done)))
    log("TOP30: " + ", ".join("%s(%.2f)" % (r["name"], r["score"]) for r in stable[:30]))

if __name__ == "__main__":
    main()
