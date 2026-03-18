cd simpledeco_vllm
VLLM_USE_PRECOMPILED=1 UV_CONCURRENCY=24 uv pip install -U -e . --no-build-isolation && uv pip install transformers==4.56.0 trl==0.22.0
cd ..
