#!/bin/bash

# ==============================================================================
# Script: measure-tcb.sh
# Description: 自動化量測 LLM TA 系統的 Trusted Computing Base (TCB) LoC。
#              包含：OP-TEE OS (Base & Delta), LLM TA, 靜態連結之 Openlibm。
# ==============================================================================

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ==============================================================================
# ⚠️ [使用者設定區] ⚠️
# 如果您的 LLM TA 引入了新的數學函數 (例如 log, tan)，請務必更新下方清單！
# ==============================================================================
OPENLIBM_SUBSET=(
    "dep/openlibm/src/s_copysign.c"
    "dep/openlibm/src/s_copysignf.c"
    "dep/openlibm/src/s_cosf.c"
    "dep/openlibm/src/e_expf.c"
    "dep/openlibm/src/s_fabsf.c"
    "dep/openlibm/src/s_floor.c"
    "dep/openlibm/src/e_powf.c"
    "dep/openlibm/src/s_scalbn.c"
    "dep/openlibm/src/s_scalbnf.c"
    "dep/openlibm/src/s_sinf.c"
    "dep/openlibm/src/e_sqrtf.c"
    "dep/openlibm/src/e_rem_pio2f.c"
    "dep/openlibm/src/k_cosf.c"
    "dep/openlibm/src/k_rem_pio2.c"
    "dep/openlibm/src/k_sinf.c"
    "dep/openlibm/src/math_private.h"
    "dep/openlibm/include/openlibm_math.h"
)
# ==============================================================================

# 預設變數
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OPTEE_OS_DIR="$HOME/optee/optee_os"
BASE_COMMIT="5aba4f"
SKIP_PROMPT=false

# 顯示使用說明
usage() {
    echo -e "Usage: $0 [--os-dir <path>] [--base-commit <commit>] [-y|--yes]"
    echo -e "Options:"
    echo -e "  -y, --yes    跳過 Openlibm 寫死清單的互動式確認警告"
    exit 1
}

# 解析參數
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --os-dir) OPTEE_OS_DIR="$2"; shift ;;
        --base-commit) BASE_COMMIT="$2"; shift ;;
        -y|--yes) SKIP_PROMPT=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$OPTEE_OS_DIR" ] || [ -z "$BASE_COMMIT" ]; then
    echo -e "${RED}錯誤：必須提供 --os-dir 與 --base-commit 參數。${NC}"
    usage
fi

echo -e "${BLUE}>>> 開始進行 TCB LoC 量測...${NC}\n"

# ==============================================================================
# 1. 量測 OP-TEE OS Base LoC
# ==============================================================================
echo -e "${YELLOW}[1/4] 量測 OP-TEE OS Baseline LoC...${NC}"
cd "$OPTEE_OS_DIR" || exit 1

rm -f /tmp/compiled_c_files.txt /tmp/compiled_s_files.txt /tmp/actual_tcb_files.txt
touch /tmp/actual_tcb_files.txt

find out/arm/core/ -type f -name "*.o" | sed 's|^out/arm/||' | sed 's/\.o$/.c/' > /tmp/compiled_c_files.txt
find out/arm/core/ -type f -name "*.o" | sed 's|^out/arm/||' | sed 's/\.o$/.S/' > /tmp/compiled_s_files.txt
find out/arm/core/ -type f -name "*.o" | sed 's|^out/arm/||' | sed 's/\.o$/.s/' >> /tmp/compiled_s_files.txt

cat /tmp/compiled_c_files.txt /tmp/compiled_s_files.txt | sort | uniq | while read -r file; do
    if [ -f "$file" ]; then
        echo "$file" >> /tmp/actual_tcb_files.txt
    fi
done

OS_BASE_LOC=$(cloc --list-file=/tmp/actual_tcb_files.txt --csv --quiet | tail -n 1 | awk -F',' '{print $5}')
echo "  -> OS Baseline LoC: $OS_BASE_LOC"

# ==============================================================================
# 2. 量測 OP-TEE OS Delta LoC (修改量)
# ==============================================================================
echo -e "${YELLOW}[2/4] 量測 OP-TEE OS 修改量 (Delta)...${NC}"
OS_ADDED=$(git diff --numstat "$BASE_COMMIT" HEAD | awk '{s+=$1} END {print s}')
OS_ADDED=${OS_ADDED:-0}
echo "  -> OS Delta Added: +$OS_ADDED"

# ==============================================================================
# 3. 量測 LLM TA LoC
# ==============================================================================
echo -e "${YELLOW}[3/4] 量測 LLM TA LoC...${NC}"
cd "$PROJECT_ROOT" || exit 1

TA_LOC=$(cloc ta/ --csv --quiet --exclude-ext=o,d,cmd | tail -n 1 | awk -F',' '{print $5}')
echo "  -> LLM TA LoC: $TA_LOC"

# ==============================================================================
# 4. 量測 Openlibm (Subset) LoC
# ==============================================================================
echo -e "${YELLOW}[4/4] 量測 Openlibm 靜態連結子集 LoC...${NC}"

# --- 互動式警告區塊 ---
if [ "$SKIP_PROMPT" = false ]; then
    echo -e ""
    echo -e "${RED}==========================================================================${NC}"
    echo -e "${RED} ⚠️  警告 (WARNING): Openlibm 靜態連結清單為腳本頂部的「寫死 (Hardcoded)」狀態！${NC}"
    echo -e "${RED}==========================================================================${NC}"
    echo -e "若您的 LLM TA 程式碼有修改 (例如引入了全新的數學函數)，"
    echo -e "請確認腳本上方 [使用者設定區] 的 OPENLIBM_SUBSET 清單是否需要更新。"
    echo -e ""
    read -p "請問您確認目前的數學依賴沒有改變，要繼續計算嗎？ (y/n): " -n 1 -r
    echo -e ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "\n${YELLOW}已中止腳本。請修改腳本最上方的 OPENLIBM_SUBSET 清單後再重新執行。${NC}"
        exit 1
    fi
    echo -e "${GREEN}確認完畢，繼續執行量測...${NC}"
fi
# ----------------------

# 將陣列內容寫入暫存檔
rm -f /tmp/openlibm_tcb_files.txt
for file in "${OPENLIBM_SUBSET[@]}"; do
    echo "$file" >> /tmp/openlibm_tcb_files.txt
done

LIB_LOC=$(cloc --list-file=/tmp/openlibm_tcb_files.txt --csv --quiet | tail -n 1 | awk -F',' '{print $5}')
echo "  -> Openlibm Subset LoC: $LIB_LOC"

# 清理暫存檔
rm -f /tmp/compiled_c_files.txt /tmp/compiled_s_files.txt /tmp/actual_tcb_files.txt /tmp/openlibm_tcb_files.txt

# ==============================================================================
# 5. 輸出論文用總結表格
# ==============================================================================
INFLATED_TCB=$((TA_LOC + LIB_LOC + OS_ADDED))

echo -e "\n${GREEN}=======================================================${NC}"
echo -e "${GREEN}                 TCB 量測結果總結                      ${NC}"
echo -e "${GREEN}=======================================================${NC}\n"

echo "| Components | Line of Code (LoC) |"
echo "| :--- | :--- |"
echo "| **Your LLM TA** | **$(printf "%'d\n" "$TA_LOC")** |"
echo "| **Openlibm (Subset)** | **$(printf "%'d\n" "$LIB_LOC")** |"
echo "| OP-TEE OS (Baseline) | $(printf "%'d\n" "$OS_BASE_LOC") + **$OS_ADDED** |"
echo "| **Inflated TCB** | **$(printf "%'d\n" "$INFLATED_TCB")** |"
echo ""
echo -e "${BLUE}提示：可直接將上方表格複製到您的 Markdown 論文草稿中。${NC}"