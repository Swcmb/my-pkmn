#!/usr/bin/env bash
# 阶段 4：WSL2 真实官方运行时验证。
# 干净目录解包现有候选包 → 叠加迁移内容（新权威库 / 当前 sdk_cg / Dragapult 牌组）→
# 逐用例独立进程运行，记录退出码（SIGABRT = 134）。
set -u
SRC=/mnt/d/pkmn-kaggle
BASE="$HOME/pkmn-migration-verify-20261020"
PKG="$BASE/pkg"
PYLIBS="$HOME/pkmn-linux-verify-20261020/pylibs"
TAR="$SRC/submission_rl_transformer2_dragapult_nodist.tar.gz"

rm -rf "$PKG"
mkdir -p "$PKG"
echo "[stage] 解包 $(basename "$TAR") -> $PKG"
tar -xzf "$TAR" -C "$PKG" || { echo "UNPACK_FAILED"; exit 1; }

cp "$SRC/EN_Card_Data.csv" "$PKG/EN_Card_Data.csv"
cp "$SRC"/sdk_cg/*.py "$PKG/sdk_cg/"
cp "$SRC/runs/evaluation/deck_search/dragapult_deck.csv" "$PKG/deck.csv"
cp "$SRC/runs/card_database_migration/20261020/wsl_stage4_cases.py" "$BASE/wsl_stage4_cases.py"

echo "[stage] 包内关键文件 sha256:"
( cd "$PKG" && sha256sum EN_Card_Data.csv deck.csv sdk_cg/sim.py main.py )

echo "[env] python/torch:"
PYTHONPATH="$PYLIBS" python3 -c "import sys,torch;print(sys.version.split()[0], torch.__version__)"

export PYTHONPATH="$PYLIBS"
export PKMN_TURN_CAP=300
RESULTS="$BASE/case_results.jsonl"
: > "$RESULTS"
for case in import_clean env_first by_path deck_request games would_ko; do
  echo "=== CASE $case ==="
  out=$(cd "$PKG" && python3 "$BASE/wsl_stage4_cases.py" "$PKG" "$PYLIBS" "$case" 2>&1)
  code=$?
  echo "exit=$code"
  echo "$out" | tail -25
  printf '{"case":"%s","exit":%d,"output":%s}\n' "$case" "$code" \
    "$(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <<< "$out")" >> "$RESULTS"
done
echo "[done] 汇总: $RESULTS"
