#!/usr/bin/env python3
"""
GPU VERIFICATION - Bulletproof Edition
Tests: CuPy, cuDF, XGBoost GPU, CatBoost GPU
Usage: ./run_gpu.sh GPU_VERIFY_QUICK.py
       ./run_gpu.sh --verify
"""
import os
import sys

# Environment setup (also done by run_gpu.sh, but safe to repeat)
os.environ.setdefault('LD_LIBRARY_PATH', '/usr/lib/wsl/lib')
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')
os.environ.setdefault('NUMBA_CUDA_DRIVER', '/usr/lib/wsl/lib/libcuda.so.1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

def test_cupy():
    """Test CuPy GPU access"""
    print("[1/4] Testing CuPy...")
    try:
        import cupy as cp
        _ = cp.cuda.Device(0).compute_capability
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props['name'].decode()
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"      GPU: {gpu_name}")
        print(f"      VRAM: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")
        print(f"      CuPy: {cp.__version__}")
        print("      PASS\n")
        return True
    except Exception as e:
        print(f"      FAIL: {e}\n")
        return False

def test_cudf():
    """Test cuDF GPU DataFrames"""
    print("[2/4] Testing cuDF...")
    try:
        import cudf
        df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = df['a'].sum()
        assert result == 6, f"Expected 6, got {result}"
        print(f"      cuDF: {cudf.__version__}")
        print("      DataFrame ops: PASS")
        print("      PASS\n")
        return True
    except Exception as e:
        print(f"      FAIL: {e}\n")
        return False

def test_xgboost():
    """Test XGBoost GPU training"""
    print("[3/4] Testing XGBoost GPU...")
    try:
        import xgboost as xgb
        import numpy as np
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        dtrain = xgb.DMatrix(X, label=y)
        params = {'tree_method': 'gpu_hist', 'device': 'cuda:0', 'verbosity': 0}
        model = xgb.train(params, dtrain, num_boost_round=10)
        print(f"      XGBoost: {xgb.__version__}")
        print("      tree_method=gpu_hist: PASS")
        print("      PASS\n")
        return True
    except Exception as e:
        print(f"      FAIL: {e}\n")
        return False

def test_catboost():
    """Test CatBoost GPU training"""
    print("[4/4] Testing CatBoost GPU...")
    try:
        from catboost import CatBoostRegressor
        import numpy as np
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        model = CatBoostRegressor(iterations=10, task_type='GPU', devices='0', verbose=0)
        model.fit(X, y)
        print("      task_type=GPU: PASS")
        print("      PASS\n")
        return True
    except Exception as e:
        print(f"      FAIL: {e}\n")
        return False

def main():
    print("\n" + "=" * 70)
    print("                    GPU VERIFICATION TEST")
    print("=" * 70 + "\n")

    results = []
    results.append(("CuPy", test_cupy()))
    results.append(("cuDF", test_cudf()))
    results.append(("XGBoost", test_xgboost()))
    results.append(("CatBoost", test_catboost()))

    print("=" * 70)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    if passed == total:
        print("         ALL GPU TESTS PASSED - SYSTEM READY")
    else:
        print(f"         {passed}/{total} TESTS PASSED")
        print("\n         Failed tests:")
        for name, result in results:
            if not result:
                print(f"           - {name}")
    print("=" * 70)

    # Final memory check
    try:
        import cupy as cp
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"\nFinal GPU Memory: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total\n")
    except:
        pass

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
