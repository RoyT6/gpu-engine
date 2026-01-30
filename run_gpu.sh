#!/bin/bash
#===============================================================================
#                    GPU RUNNER v2.0 - BULLETPROOF EDITION
#              Reads conda path from config - No guessing, no failures
#===============================================================================
#  Usage: ./run_gpu.sh your_script.py [args]
#  Commands: --help, --verify, --reset, --status
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/gpu_config.json"
CB_STATE="$SCRIPT_DIR/.circuit_breaker.json"
LOG_DIR="$SCRIPT_DIR/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

#===============================================================================
# DEPENDENCY CHECK - Fail fast with clear instructions
#===============================================================================
check_dependencies() {
    local missing=()

    if ! command -v jq &>/dev/null; then
        missing+=("jq")
    fi

    if ! command -v bc &>/dev/null; then
        missing+=("bc")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "${RED}===============================================================================${NC}"
        echo -e "${RED}                    MISSING DEPENDENCIES${NC}"
        echo -e "${RED}===============================================================================${NC}"
        echo ""
        for dep in "${missing[@]}"; do
            echo -e "  ${RED}X${NC} $dep is not installed"
        done
        echo ""
        echo -e "${YELLOW}Install with:${NC}"
        echo "  sudo apt-get update && sudo apt-get install -y ${missing[*]}"
        echo ""
        exit 1
    fi
}

check_dependencies

#===============================================================================
# CONFIG VALIDATION - Config file must exist and be valid
#===============================================================================
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Validate JSON
if ! jq empty "$CONFIG_FILE" 2>/dev/null; then
    echo -e "${RED}ERROR: Invalid JSON in config file${NC}"
    exit 1
fi

#===============================================================================
# READ CONFIG - All paths come from config, no guessing
#===============================================================================
CONDA_BASE=$(jq -r '.conda.base_path' "$CONFIG_FILE")
DEFAULT_ENV=$(jq -r '.conda.default_environment' "$CONFIG_FILE")
FALLBACK_ENVS=$(jq -r '.conda.fallback_environments[]?' "$CONFIG_FILE" 2>/dev/null)

#===============================================================================
# CIRCUIT BREAKER
#===============================================================================
init_circuit_breaker() {
    if [[ ! -f "$CB_STATE" ]]; then
        echo '{"state":"CLOSED","failures":0,"last_failure":""}' > "$CB_STATE"
    fi
}

get_circuit_state() {
    init_circuit_breaker
    jq -r '.state // "CLOSED"' "$CB_STATE" 2>/dev/null || echo "CLOSED"
}

record_failure() {
    local reason="$1"
    init_circuit_breaker
    local failures=$(jq -r '.failures // 0' "$CB_STATE")
    failures=$((failures + 1))
    local state="CLOSED"
    [[ $failures -ge 2 ]] && state="HALF_OPEN"
    [[ $failures -ge 3 ]] && state="OPEN"
    echo "{\"state\":\"$state\",\"failures\":$failures,\"last_failure\":\"$reason\"}" > "$CB_STATE"
    echo "$state"
}

record_success() {
    echo '{"state":"CLOSED","failures":0,"last_failure":""}' > "$CB_STATE"
}

reset_circuit_breaker() {
    echo '{"state":"CLOSED","failures":0,"last_failure":""}' > "$CB_STATE"
    echo -e "${GREEN}Circuit breaker reset to CLOSED${NC}"
}

#===============================================================================
# HEADER
#===============================================================================
print_header() {
    echo ""
    echo -e "${CYAN}===============================================================================${NC}"
    echo -e "${CYAN}                    GPU RUNNER v2.0 - BULLETPROOF EDITION${NC}"
    echo -e "${CYAN}===============================================================================${NC}"
}

#===============================================================================
# COMMAND HANDLERS
#===============================================================================
case "${1:-}" in
    --help|-h)
        print_header
        echo ""
        echo "Usage: ./run_gpu.sh <script.py> [args]"
        echo ""
        echo "Commands:"
        echo "  --help, -h      Show this help"
        echo "  --verify        Run GPU verification test"
        echo "  --reset         Reset circuit breaker"
        echo "  --status        Show system status"
        echo "  --config        Show configuration"
        echo ""
        exit 0
        ;;
    --reset)
        print_header
        reset_circuit_breaker
        exit 0
        ;;
    --status)
        print_header
        echo ""
        echo -e "${BLUE}Circuit Breaker:${NC}"
        [[ -f "$CB_STATE" ]] && jq '.' "$CB_STATE" || echo "Not initialized"
        echo ""
        echo -e "${BLUE}Conda:${NC}"
        echo "  Base: $CONDA_BASE"
        echo "  Environment: $DEFAULT_ENV"
        echo ""
        exit 0
        ;;
    --config)
        print_header
        echo ""
        jq '.' "$CONFIG_FILE"
        echo ""
        exit 0
        ;;
    --verify)
        # Will run GPU_VERIFY_QUICK.py below
        set -- "GPU_VERIFY_QUICK.py"
        ;;
esac

print_header

#===============================================================================
# CIRCUIT BREAKER CHECK
#===============================================================================
init_circuit_breaker
CB=$(get_circuit_state)

if [[ "$CB" == "OPEN" ]]; then
    echo ""
    echo -e "${RED}===============================================================================${NC}"
    echo -e "${RED}          CIRCUIT BREAKER OPEN - EXECUTION BLOCKED${NC}"
    echo -e "${RED}===============================================================================${NC}"
    jq '.' "$CB_STATE"
    echo ""
    echo -e "${YELLOW}Run './run_gpu.sh --reset' to reset${NC}"
    exit 1
fi

#===============================================================================
# SCRIPT VALIDATION
#===============================================================================
if [[ -z "${1:-}" ]]; then
    echo -e "${RED}Error: No Python script specified${NC}"
    echo "Usage: ./run_gpu.sh your_script.py [args]"
    exit 1
fi

SCRIPT="$1"
shift

if [[ ! -f "$SCRIPT" ]]; then
    echo -e "${RED}Error: Script not found: $SCRIPT${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Script:${NC} $SCRIPT"

#===============================================================================
# ENVIRONMENT VARIABLES - From config
#===============================================================================
echo ""
echo -e "${BLUE}[1/4] Setting environment variables...${NC}"

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export NUMBA_CUDA_DRIVER=/usr/lib/wsl/lib/libcuda.so.1
export CUDA_VISIBLE_DEVICES=0
export CUDF_SPILL=on
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

echo -e "  ${GREEN}OK${NC} Environment variables set"

#===============================================================================
# CONDA ACTIVATION - From config, with validation
#===============================================================================
echo ""
echo -e "${BLUE}[2/4] Activating conda environment...${NC}"

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
    echo -e "${RED}===============================================================================${NC}"
    echo -e "${RED}          CONDA NOT FOUND AT CONFIGURED PATH${NC}"
    echo -e "${RED}===============================================================================${NC}"
    echo ""
    echo -e "Expected: ${YELLOW}$CONDA_SH${NC}"
    echo ""
    echo "Fix gpu_config.json conda.base_path to point to your conda installation."
    echo ""
    echo "Common locations:"
    echo "  ~/miniforge3"
    echo "  ~/miniconda3"
    echo "  ~/anaconda3"
    echo ""
    record_failure "Conda not found at $CONDA_BASE"
    exit 1
fi

source "$CONDA_SH"

# Find working environment
ACTIVE_ENV=""
for env in "$DEFAULT_ENV" $FALLBACK_ENVS; do
    if [[ -d "$CONDA_BASE/envs/$env" ]]; then
        ACTIVE_ENV="$env"
        break
    fi
done

if [[ -z "$ACTIVE_ENV" ]]; then
    echo -e "${RED}===============================================================================${NC}"
    echo -e "${RED}          NO VALID CONDA ENVIRONMENT FOUND${NC}"
    echo -e "${RED}===============================================================================${NC}"
    echo ""
    echo "Searched for: $DEFAULT_ENV $FALLBACK_ENVS"
    echo ""
    echo "Available environments:"
    ls "$CONDA_BASE/envs/" 2>/dev/null || echo "  (none)"
    echo ""
    record_failure "No valid conda environment"
    exit 1
fi

conda activate "$ACTIVE_ENV"
echo -e "  ${GREEN}OK${NC} Activated: $ACTIVE_ENV"

#===============================================================================
# GPU VERIFICATION
#===============================================================================
echo ""
echo -e "${BLUE}[3/4] Verifying GPU access...${NC}"

GPU_CHECK=$(python -c "
import json
try:
    import cupy as cp
    _ = cp.cuda.Device(0).compute_capability
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name'].decode()
    mem_free, mem_total = cp.cuda.runtime.memGetInfo()
    print(json.dumps({
        'status': 'OK',
        'gpu': name,
        'vram_free_gb': round(mem_free/1e9, 1),
        'vram_total_gb': round(mem_total/1e9, 1),
        'memory_used': mem_total - mem_free
    }))
except Exception as e:
    print(json.dumps({'status': 'FAIL', 'error': str(e)}))
" 2>/dev/null) || GPU_CHECK='{"status":"FAIL","error":"Python execution failed"}'

GPU_STATUS=$(echo "$GPU_CHECK" | jq -r '.status')

if [[ "$GPU_STATUS" != "OK" ]]; then
    GPU_ERROR=$(echo "$GPU_CHECK" | jq -r '.error // "Unknown error"')
    echo -e "${RED}===============================================================================${NC}"
    echo -e "${RED}          GPU VERIFICATION FAILED${NC}"
    echo -e "${RED}===============================================================================${NC}"
    echo ""
    echo -e "Error: $GPU_ERROR"
    echo ""
    echo -e "${RED}NO CPU FALLBACK - GPU IS MANDATORY${NC}"
    echo ""
    record_failure "GPU verification: $GPU_ERROR"
    exit 1
fi

GPU_NAME=$(echo "$GPU_CHECK" | jq -r '.gpu')
GPU_VRAM_FREE=$(echo "$GPU_CHECK" | jq -r '.vram_free_gb')
GPU_VRAM_TOTAL=$(echo "$GPU_CHECK" | jq -r '.vram_total_gb')
GPU_MEM_BEFORE=$(echo "$GPU_CHECK" | jq -r '.memory_used')

echo -e "  ${GREEN}OK${NC} GPU: $GPU_NAME"
echo -e "  ${GREEN}OK${NC} VRAM: ${GPU_VRAM_FREE}GB free / ${GPU_VRAM_TOTAL}GB total"

#===============================================================================
# RUN SCRIPT
#===============================================================================
echo ""
echo -e "${BLUE}[4/4] Running script...${NC}"
echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}                         EXECUTING: $SCRIPT${NC}"
echo -e "${GREEN}===============================================================================${NC}"
echo ""

START_TIME=$(date +%s.%N)
EXIT_CODE=0
python "$SCRIPT" "$@" || EXIT_CODE=$?
END_TIME=$(date +%s.%N)
EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""

#===============================================================================
# POST-EXECUTION VALIDATION
#===============================================================================
GPU_AFTER=$(python -c "
import json
try:
    import cupy as cp
    mem_free, mem_total = cp.cuda.runtime.memGetInfo()
    print(json.dumps({'memory_used': mem_total - mem_free}))
except:
    print(json.dumps({'memory_used': 0}))
" 2>/dev/null) || GPU_AFTER='{"memory_used":0}'

GPU_MEM_AFTER=$(echo "$GPU_AFTER" | jq -r '.memory_used')
GPU_MEM_DELTA=$((GPU_MEM_AFTER - GPU_MEM_BEFORE))

#===============================================================================
# RESULT
#===============================================================================
echo "==============================================================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "                    ${GREEN}COMPLETED SUCCESSFULLY${NC}"
    record_success
else
    echo -e "                    ${RED}FAILED (exit code: $EXIT_CODE)${NC}"
    record_failure "Script failed with exit code $EXIT_CODE"
fi
echo "==============================================================================="
echo ""
echo -e "  Execution time: ${EXEC_TIME}s"
echo -e "  GPU memory delta: $((GPU_MEM_DELTA / 1024 / 1024))MB"
echo ""

exit $EXIT_CODE
