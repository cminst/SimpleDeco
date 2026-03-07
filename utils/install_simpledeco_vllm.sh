cd simpledeco_vllm
UV_CONCURRENCY=24 uv pip install -U -e . --no-build-isolation && uv pip install transformers==4.56.0 trl==0.22.0
cd ..
