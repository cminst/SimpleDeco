cd simpledeco_vllm
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/0.10.2/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl \
    uv pip install -U -e . --no-build-isolation --system --torch-backend=cu128 \
    && uv pip install transformers==4.56.0 trl==0.22.0 --system --torch-backend=cu128
cd ..
