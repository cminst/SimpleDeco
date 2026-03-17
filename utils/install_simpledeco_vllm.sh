cd simpledeco_vllm
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/0.10.2/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl UV_CONCURRENCY=24 uv pip install -U -e . --no-build-isolation && uv pip install transformers==4.56.0 trl==0.22.0
cd ..
