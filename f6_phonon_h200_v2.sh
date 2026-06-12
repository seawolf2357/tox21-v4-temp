#!/bin/bash
# F6 포논 계산 v2 — H200 (conda QE 설치 + 단일 프로세스, buffer overflow 우회)
# 원인: /usr/bin/pw.x (Ubuntu 패키지) = buffer overflow SIGABRT
# 해결: conda-forge QE (자체 빌드) + mpirun 없이 OMP 단일 프로세스 (H100 방식)
WORKDIR="/home/work/vidraft/f6_phonon"
LOG="$WORKDIR/f6.log"
mkdir -p "$WORKDIR/tmp_F6"
cd "$WORKDIR"

echo "=== F6 Phonon v2 (conda QE) on H200 ===" | tee $LOG
date | tee -a $LOG

# ─── conda QE 준비 ───────────────────────────────────────────────────
CONDA_SH="/home/work/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda.sh 없음 ($CONDA_SH)" | tee -a $LOG; exit 1
fi
source "$CONDA_SH"

if ! conda env list | grep -qE "^qe[[:space:]]"; then
    echo "[INSTALL] conda QE 설치 중 (~5-10분, conda-forge)..." | tee -a $LOG
    conda create -n qe -c conda-forge "qe>=7.0" -y 2>&1 | tail -15 | tee -a $LOG
fi
conda activate qe
QE_BIN="$(dirname "$(which pw.x 2>/dev/null)")"
if [ ! -x "$QE_BIN/pw.x" ] || [ ! -x "$QE_BIN/ph.x" ]; then
    echo "ERROR: conda QE 설치 실패 — pw.x/ph.x 없음 (QE_BIN=$QE_BIN)" | tee -a $LOG
    exit 1
fi
echo "[OK] conda QE: $QE_BIN" | tee -a $LOG
"$QE_BIN/pw.x" --version 2>/dev/null | head -1 | tee -a $LOG

# ─── 슈도포텐셜 (검증된 GitHub 사본) ─────────────────────────────────
PSEUDO_DIR="$WORKDIR/pseudo"
mkdir -p "$PSEUDO_DIR"
for f in La.paw.z_11.atompaw.wentzcovitch.v1.2.upf ni_pbe_v1.4.uspp.F.UPF O.pbe-n-kjpaw_psl.0.1.UPF f_pbe_v1.4.uspp.F.UPF; do
    if [ ! -s "$PSEUDO_DIR/$f" ]; then
        wget -q "https://raw.githubusercontent.com/seawolf2357/tox21-v4-temp/main/pseudo/$f" -O "$PSEUDO_DIR/$f"
    fi
    if ! head -c 200 "$PSEUDO_DIR/$f" 2>/dev/null | grep -qi "UPF\|PP_INFO\|<UPF"; then
        echo "ERROR: $f invalid UPF" | tee -a $LOG; exit 1
    fi
done
echo "[OK] 슈도포텐셜 4종 확인" | tee -a $LOG

# ─── SCF 입력 (H100 검증값: ecut 40/320, beta 0.20, conv 1e-7) ───────
cat > "$WORKDIR/F6_scf.in" << 'SCFINPUT'
&CONTROL
 calculation = "scf"
 prefix      = "F6"
 outdir      = "./tmp_F6"
 pseudo_dir  = "./pseudo"
 verbosity   = "high"
/
&SYSTEM
 ibrav = 0, nat = 24, ntyp = 4
 ecutwfc = 40, ecutrho = 320
 occupations = "smearing", smearing = "mp", degauss = 0.02
 nspin = 2
 starting_magnetization(2) = 0.3
/
&ELECTRONS
 conv_thr         = 1.0d-7
 mixing_mode      = "local-TF"
 mixing_beta      = 0.20
 electron_maxstep = 300
 diagonalization  = "david"
/
ATOMIC_SPECIES
La  138.9050  La.paw.z_11.atompaw.wentzcovitch.v1.2.upf
Ni   58.6934  ni_pbe_v1.4.uspp.F.UPF
O    15.9990  O.pbe-n-kjpaw_psl.0.1.UPF
F    18.9980  f_pbe_v1.4.uspp.F.UPF
CELL_PARAMETERS angstrom
3.813979  0.0       0.0
0.0       3.813979  0.0
0.0       0.0       20.267919
ATOMIC_POSITIONS crystal
La  0.000000  0.000000  0.500000
La  0.500000  0.500000  0.000000
La  0.000000  0.000000  0.320966
La  0.500000  0.500000  0.820967
La  0.000000  0.000000  0.679034
La  0.500000  0.500000  0.179033
Ni  0.000000  0.000000  0.098364
Ni  0.500000  0.500000  0.598373
Ni  0.000000  0.000000  0.901636
Ni  0.500000  0.500000  0.401627
F   0.000000  0.000000  0.000000
F   0.500000  0.500000  0.500000
F   0.000000  0.000000  0.204552
F   0.500000  0.500000  0.704558
F   0.000000  0.000000  0.795448
F   0.500000  0.500000  0.295442
O   0.000000  0.500000  0.097396
O   0.500000  0.000000  0.597399
O   0.500000  0.000000  0.097396
O   0.000000  0.500000  0.597399
O   0.000000  0.500000  0.902604
O   0.500000  0.000000  0.402601
O   0.500000  0.000000  0.902604
O   0.000000  0.500000  0.402601
K_POINTS automatic
6 6 1 0 0 0
SCFINPUT

# ─── 포논 입력 (Gamma point electron-phonon) ────────────────────────
cat > "$WORKDIR/F6_ph.in" << 'PHINPUT'
Electron-phonon F6
&INPUTPH
 prefix       = "F6"
 outdir       = "./tmp_F6"
 fildyn       = "F6.dyn"
 ldisp        = .false.
 electron_phonon = "interpolated"
 el_ph_sigma  = 0.005
 el_ph_nsigma = 10
 tr2_ph       = 1.0d-12
/
0.0 0.0 0.0
PHINPUT

# ─── Tc 계산 스크립트 ────────────────────────────────────────────────
cat > "$WORKDIR/calc_tc.py" << 'PYEOF'
import re, sys, math
def allen_dynes(lam, w_K, mu=0.10):
    if lam <= mu: return 0.0
    return (w_K/1.2)*math.exp(-1.04*(1+lam)/(lam-mu*(1+0.62*lam)))
txt = open(sys.argv[1]).read()
lam = re.search(r"lambda\s*=\s*([0-9.]+)", txt)
omg = re.search(r"omega_log\s*=\s*([0-9.]+)\s*(THz|K)", txt)
if not lam or not omg:
    for l in txt.split("\n"):
        if "lambda" in l.lower() or "omega" in l.lower(): print(l.strip())
    print("ERROR: lambda/omega_log not found"); sys.exit(1)
lam=float(lam.group(1)); w=float(omg.group(1)); u=omg.group(2)
wK = w*47.99 if u=="THz" else w
print(f"\n{'='*46}\n  lambda    = {lam:.4f}\n  omega_log = {w:.2f} {u} ({wK:.1f} K)")
print(f"  Tc(mu*0.10) = {allen_dynes(lam,wK,0.10):.1f} K")
print(f"  Tc(mu*0.13) = {allen_dynes(lam,wK,0.13):.1f} K\n{'='*46}")
PYEOF

# ─── 단일 프로세스 SCF (OMP, mpirun 없음 → buffer overflow 우회) ─────
NCPU=$(nproc); OMP=$((NCPU<16?NCPU:16))
export OMP_NUM_THREADS=$OMP
echo "[SCF] pw.x 단일프로세스 OMP=$OMP (mpirun 없음)..." | tee -a $LOG
"$QE_BIN/pw.x" -input F6_scf.in > F6_scf.out 2>&1
if ! grep -q "convergence has been achieved" F6_scf.out; then
    echo "ERROR: SCF 미수렴/실패" | tee -a $LOG
    tail -25 F6_scf.out | tee -a $LOG
    exit 1
fi
echo "[OK] SCF 수렴!" | tee -a $LOG
grep "total energy" F6_scf.out | tail -2 | tee -a $LOG

# ─── 포논 (electron-phonon) ──────────────────────────────────────────
echo "[PH] ph.x electron-phonon (단일프로세스)..." | tee -a $LOG
"$QE_BIN/ph.x" -input F6_ph.in > F6_ph.out 2>&1
echo "[OK] ph.x 완료" | tee -a $LOG

# ─── Allen-Dynes Tc ──────────────────────────────────────────────────
echo "" | tee -a $LOG
python3 "$WORKDIR/calc_tc.py" F6_ph.out 2>&1 | tee -a $LOG
echo "=== DONE ($(date)) ===" | tee -a $LOG
