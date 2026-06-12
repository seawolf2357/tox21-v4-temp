#!/bin/bash
# F6 포논 계산 — H200 실행 스크립트
# 실행: bash /home/work/vidraft/f6_phonon/run_f6.sh
set -e
WORKDIR="/home/work/vidraft/f6_phonon"
LOG="$WORKDIR/f6_phonon.log"
mkdir -p "$WORKDIR/tmp_F6"
cd "$WORKDIR"

echo "=== F6 Phonon Calculation on H200 ===" | tee $LOG
date | tee -a $LOG

# QE 경로 탐색
QE_BIN=""
for p in \
    "/home/work/.local/bin" \
    "/usr/local/bin" \
    "/opt/conda/bin" \
    "$HOME/miniconda3/envs/qe/bin" \
    "$HOME/miniconda3/bin"; do
    if [ -x "$p/pw.x" ]; then QE_BIN="$p"; break; fi
done

if [ -z "$QE_BIN" ]; then
    echo "[INSTALL] QE not found — installing via conda..." | tee -a $LOG
    conda create -n qe -c conda-forge qe -y 2>&1 | tail -5
    QE_BIN="$HOME/miniconda3/envs/qe/bin"
fi
echo "[OK] QE bin: $QE_BIN" | tee -a $LOG

# 슈도포텐셜 확인
PSEUDO_DIR="$WORKDIR/pseudo"
mkdir -p "$PSEUDO_DIR"
for f in La.paw.z_11.atompaw.wentzcovitch.v1.2.upf ni_pbe_v1.4.uspp.F.UPF O.pbe-n-kjpaw_psl.0.1.UPF f_pbe_v1.4.uspp.F.UPF; do
    if [ ! -f "$PSEUDO_DIR/$f" ]; then
        echo "[DL] $f" | tee -a $LOG
        wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/$f" -O "$PSEUDO_DIR/$f" || \
        wget -q "https://www.quantum-espresso.org/upf_files/$f" -O "$PSEUDO_DIR/$f" || \
        echo "WARNING: $f download failed — place manually in $PSEUDO_DIR" | tee -a $LOG
    fi
done

# ─── SCF 입력 파일 생성 ───────────────────────────────────────────────
cat > "$WORKDIR/F6_scf.in" << 'SCFINPUT'
&CONTROL
 calculation = "scf"
 prefix      = "F6"
 outdir      = "./tmp_F6"
 pseudo_dir  = "./pseudo"
/
&SYSTEM
 ibrav = 0, nat = 24, ntyp = 4
 ecutwfc = 50, ecutrho = 400
 occupations = "smearing", smearing = "mp", degauss = 0.02
 nspin = 2
 starting_magnetization(2) = 0.3
/
&ELECTRONS
 conv_thr         = 1.0d-8
 mixing_mode      = "local-TF"
 mixing_beta      = 0.30
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
8 8 1 0 0 0
SCFINPUT

# ─── 포논(elph) 입력 파일 ────────────────────────────────────────────
cat > "$WORKDIR/F6_ph.in" << 'PHINPUT'
Electron-phonon F6
&INPUTPH
 prefix       = "F6"
 outdir       = "./tmp_F6"
 fildyn       = "F6.dyn"
 ldisp        = .false.
 qplot        = .false.
 electron_phonon = "interpolated"
 el_ph_sigma  = 0.005
 el_ph_nsigma = 10
 tr2_ph       = 1.0d-14
/
0.0 0.0 0.0
PHINPUT

# ─── Tc 계산 파이썬 스크립트 ─────────────────────────────────────────
cat > "$WORKDIR/calc_tc.py" << 'PYEOF'
#!/usr/bin/env python3
"""Allen-Dynes Tc from ph.x elph output"""
import re, sys, math

def allen_dynes(lam, omega_log_K, mu_star=0.10):
    if lam <= mu_star:
        return 0.0
    return (omega_log_K / 1.2) * math.exp(
        -1.04 * (1 + lam) / (lam - mu_star * (1 + 0.62 * lam))
    )

ph_out = sys.argv[1] if len(sys.argv) > 1 else "F6_ph.out"
text = open(ph_out).read()

lam_m = re.search(r"lambda\s*=\s*([0-9.]+)", text)
omg_m = re.search(r"omega_log\s*=\s*([0-9.]+)\s*(THz|K)", text)

if not lam_m or not omg_m:
    # 찾지 못하면 grep으로 직접 찾기
    for line in text.split('\n'):
        if 'lambda' in line.lower():
            print("LAMBDA LINE:", line.strip())
        if 'omega' in line.lower():
            print("OMEGA LINE:", line.strip())
    print("ERROR: lambda or omega_log not found in", ph_out)
    sys.exit(1)

lam = float(lam_m.group(1))
omg_raw = float(omg_m.group(1))
unit = omg_m.group(2)
omg_K = omg_raw * 47.99 if unit == "THz" else omg_raw  # 1 THz = 47.99 K

tc_10  = allen_dynes(lam, omg_K, mu_star=0.10)
tc_13  = allen_dynes(lam, omg_K, mu_star=0.13)

print(f"\n{'='*50}")
print(f"  lambda       = {lam:.4f}")
print(f"  omega_log    = {omg_raw:.2f} {unit}  ({omg_K:.1f} K)")
print(f"  Tc (mu*=0.10)= {tc_10:.1f} K  ({tc_10-273.15:.1f} C)")
print(f"  Tc (mu*=0.13)= {tc_13:.1f} K  ({tc_13-273.15:.1f} C)")
print(f"{'='*50}\n")
PYEOF

# ─── SCF 실행 ────────────────────────────────────────────────────────
NCPU=$(nproc)
NPAR=$((NCPU < 8 ? NCPU : 8))
echo "[SCF] mpirun -np $NPAR pw.x ..." | tee -a $LOG
cd "$WORKDIR"
mpirun -np $NPAR "$QE_BIN/pw.x" -input F6_scf.in > F6_scf.out 2>&1 &
SCF_PID=$!
echo "SCF PID: $SCF_PID" | tee -a $LOG

# 수렴 대기 + 진행 모니터
echo "=== SCF 모니터링 (30초 간격) ===" | tee -a $LOG
while kill -0 $SCF_PID 2>/dev/null; do
    sleep 30
    tail -3 F6_scf.out 2>/dev/null | grep -E "total energy|iteration|convergence" | tail -1 | tee -a $LOG
done
wait $SCF_PID
SCF_EXIT=$?

if [ $SCF_EXIT -ne 0 ]; then
    echo "ERROR: SCF failed (exit $SCF_EXIT)" | tee -a $LOG
    tail -20 F6_scf.out | tee -a $LOG
    exit 1
fi

# 수렴 확인
if ! grep -q "convergence has been achieved" F6_scf.out; then
    echo "ERROR: SCF did not converge" | tee -a $LOG
    exit 1
fi
echo "[OK] SCF converged!" | tee -a $LOG
grep "total energy" F6_scf.out | tail -3 | tee -a $LOG

# ─── 포논 (electron-phonon) 실행 ─────────────────────────────────────
echo "[PH] mpirun -np $NPAR ph.x ..." | tee -a $LOG
mpirun -np $NPAR "$QE_BIN/ph.x" -input F6_ph.in > F6_ph.out 2>&1
echo "[OK] ph.x done" | tee -a $LOG

# ─── Allen-Dynes Tc 계산 ──────────────────────────────────────────────
echo "" | tee -a $LOG
python3 "$WORKDIR/calc_tc.py" F6_ph.out | tee -a $LOG
echo "=== DONE. Log: $LOG ==="
