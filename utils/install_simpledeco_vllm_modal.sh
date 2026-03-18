cd simpledeco_vllm
VLLM_USE_PRECOMPILED=1 \
    uv pip install -U -e . --no-build-isolation --system --torch-backend=cu128 \
    && uv pip install transformers==4.56.0 trl==0.22.0 --system --torch-backend=cu128
cd ..
