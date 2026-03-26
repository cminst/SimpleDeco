cd simpledeco_vllm
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/c9d838fc338db9a5a23cb3906d17c47423c4c9e4/vllm-0.17.2rc1.dev71%2Bgc9d838fc3-cp38-abi3-manylinux_2_31_x86_64.whl" \
    uv pip install -U -e . --no-build-isolation --system --torch-backend=cu128 \
    && uv pip install transformers==4.56.0 trl==0.22.0 --system --torch-backend=cu128
cd ..
