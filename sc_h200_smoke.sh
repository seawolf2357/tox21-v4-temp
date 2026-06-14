#!/bin/bash
# H200 QE smoke 테스트 — QE 설치 + LiB2 vc-relax 1개로 buffer overflow 확인
# 성공하면 나머지 4개(NaB2 CaB2 AlB2 SrB2) 자동 추가
set -u
L=/home/work/vidraft/sc_h200
mkdir -p $L; cd $L
LOG=$L/run.log
echo "START $(date)" > $LOG

# 1. QE 설치 (conda-forge, 플러그인 우회) — 시스템 /usr/bin/pw.x는 buffer overflow, conda QE로 회피 시도
source /home/work/miniconda3/etc/profile.d/conda.sh 2>/dev/null
QE=/home/work/miniconda3/envs/qe/bin
if [ ! -x "$QE/pw.x" ]; then
  echo "[install] conda QE 설치중 (~10분, NO_PLUGINS)..." | tee -a $LOG
  CONDA_NO_PLUGINS=true conda create -n qe -c conda-forge "qe>=7.2" -y >> $LOG 2>&1
fi
if [ ! -x "$QE/pw.x" ]; then echo "CONDA_QE_INSTALL_FAIL — H100 우회 필요" | tee -a $LOG; exit 1; fi
export LD_LIBRARY_PATH=$QE/../lib:${LD_LIBRARY_PATH:-}
echo "QE=$QE (conda)" | tee -a $LOG
"$QE/pw.x" --version 2>&1 | head -1 | tee -a $LOG

# 2. 슈도 다운로드
PD=$L/pseudo; mkdir -p $PD; cd $PD
BASE=https://pseudopotentials.quantum-espresso.org/upf_files
for f in Li.pbe-s-kjpaw_psl.1.0.0.UPF Na.pbe-spn-kjpaw_psl.1.0.0.UPF Ca.pbe-spn-kjpaw_psl.1.0.0.UPF Al.pbe-n-kjpaw_psl.1.0.0.UPF Sr.pbe-spn-kjpaw_psl.1.0.0.UPF B.pbe-n-kjpaw_psl.1.0.0.UPF; do
  [ -s "$f" ] || wget -q --timeout=30 "$BASE/$f"
done
echo "슈도 $(ls *.UPF 2>/dev/null | wc -l)개" | tee -a $LOG

# 3. vc-relax 체인 함수 (ibrav=4 유지 = H100 abort 회피)
run_cand() {
  local P=$1 MEL=$2 MUPF=$3 CD1=$4 MASS=$5
  local W=$L/$P; mkdir -p $W/tmp; cd $W
  cat > vcr.in <<EOF
&CONTROL
 calculation='vc-relax'
 prefix='$P'
 outdir='./tmp'
 pseudo_dir='$PD'
 nstep=60
 forc_conv_thr=1.0d-4
/
&SYSTEM
 ibrav=4, celldm(1)=$CD1, celldm(3)=1.06
 nat=3, ntyp=2
 ecutwfc=40, ecutrho=320
 occupations='smearing', smearing='mp', degauss=0.03
/
&ELECTRONS
 conv_thr=1.0d-7, mixing_beta=0.7
/
&IONS
/
&CELL
 cell_dofree='all'
/
ATOMIC_SPECIES
$MEL $MASS $MUPF
B 10.811 B.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
$MEL 0.000000 0.000000 0.000000
B 0.333333 0.666667 0.500000
B 0.666667 0.333333 0.500000
K_POINTS automatic
12 12 12 0 0 0
EOF
  "$QE/mpirun" --allow-run-as-root -np 20 "$QE/pw.x" -in vcr.in > vcr.out 2>&1
  if grep -q "JOB DONE" vcr.out; then echo "$P VCR_DONE $(date)" >> $LOG
  else echo "$P VCR_FAIL/STOP $(date)" >> $LOG; fi
}

# 4. LiB2 smoke
echo "[smoke] LiB2 vc-relax 시작 $(date)" | tee -a $LOG
run_cand LiB2 Li Li.pbe-s-kjpaw_psl.1.0.0.UPF 5.65 6.94

# 5. buffer overflow / 작동 판정
cd $L/LiB2
if grep -qiE "buffer overflow|Backtrace|Aborted|core dump|SIGABRT" vcr.out; then
  echo "❌ BUFFER_OVERFLOW 감지 — H200 QE 불가, H100 우회 권장" | tee -a $LOG
elif grep -q "JOB DONE" vcr.out || grep -q "bfgs" vcr.out; then
  echo "✅ H200 QE 작동 확인! 나머지 4개 추가 시작" | tee -a $LOG
  ( run_cand NaB2 Na Na.pbe-spn-kjpaw_psl.1.0.0.UPF 6.30 22.99 ) &
  ( run_cand CaB2 Ca Ca.pbe-spn-kjpaw_psl.1.0.0.UPF 6.50 40.08 ) &
  ( run_cand AlB2 Al Al.pbe-n-kjpaw_psl.1.0.0.UPF 5.67 26.98 ) &
  ( run_cand SrB2 Sr Sr.pbe-spn-kjpaw_psl.1.0.0.UPF 7.00 87.62 ) &
  wait
  echo "ALL 5 vc-relax 시도 완료 $(date)" | tee -a $LOG
else
  echo "⏳ LiB2 미완/기타: $(tail -3 vcr.out)" | tee -a $LOG
fi
echo "SMOKE_END $(date)" | tee -a $LOG
