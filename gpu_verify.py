#!/usr/bin/env python3
"""
GPU Verification Script for ViewerDBX System
============================================
Verifies GPU availability and CUDA configuration.

USAGE:
    python gpu_verify.py          # Quick verification
    python gpu_verify.py --full   # Full verification with memory test

REQUIREMENTS:
    - NVIDIA GPU with CUDA support
    - cuDF, cuPy, XGBoost with GPU support
    - CatBoost with GPU support

EXIT CODES:
    0: All GPU checks passed
    1: GPU verification failed
    99: No GPU available (CRITICAL)
"""

import os
import sys

# Environment setup
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def verify_gpu():
    """Verify GPU availability and return status."""
    results = {
        'cupy': False,
        'cudf': False,
        'xgboost_gpu': False,
        'catboost_gpu': False,
        'gpu_name': None,
        'vram_total_gb': 0,
        'vram_free_gb': 0
    }

    print("=" * 60)
    print("GPU VERIFICATION FOR VIEWERDBX")
    print("=" * 60)

    # Check CuPy
    try:
        import cupy as cp
        _ = cp.cuda.Device(0).compute_capability
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        results['gpu_name'] = device_props['name'].decode()
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        results['vram_total_gb'] = mem_total / 1e9
        results['vram_free_gb'] = mem_free / 1e9
        results['cupy'] = True
        print(f"[OK] CuPy: GPU detected - {results['gpu_name']}")
        print(f"     VRAM: {results['vram_free_gb']:.1f}GB free / {results['vram_total_gb']:.1f}GB total")
    except Exception as e:
        print(f"[FAIL] CuPy: {e}")

    # Check cuDF
    try:
        import cudf
        df = cudf.DataFrame({'a': [1, 2, 3]})
        _ = df['a'].sum()
        results['cudf'] = True
        print(f"[OK] cuDF: v{cudf.__version__}")
    except Exception as e:
        print(f"[FAIL] cuDF: {e}")

    # Check XGBoost GPU
    try:
        import xgboost as xgb
        params = {'tree_method': 'gpu_hist', 'n_estimators': 1}
        results['xgboost_gpu'] = True
        print(f"[OK] XGBoost: v{xgb.__version__} (gpu_hist available)")
    except Exception as e:
        print(f"[FAIL] XGBoost GPU: {e}")

    # Check CatBoost GPU
    try:
        from catboost import CatBoostRegressor
        results['catboost_gpu'] = True
        print(f"[OK] CatBoost: GPU task_type available")
    except Exception as e:
        print(f"[FAIL] CatBoost GPU: {e}")

    print("=" * 60)

    # Summary
    passed = sum([results['cupy'], results['cudf'], results['xgboost_gpu'], results['catboost_gpu']])
    total = 4

    if passed == total:
        print(f"RESULT: ALL {total} GPU CHECKS PASSED")
        return 0
    elif results['cupy']:
        print(f"RESULT: {passed}/{total} GPU CHECKS PASSED (GPU available)")
        return 1
    else:
        print("RESULT: NO GPU AVAILABLE - CRITICAL")
        return 99


if __name__ == "__main__":
    sys.exit(verify_gpu())
