cd simpledeco_vllm
VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm-0.17.1-cp311-abi3-manylinux_2_31_x86_64.whl" \
    uv pip install -U -e . --no-build-isolation --system --torch-backend=cu128 \
    && uv pip install transformers==4.56.0 trl==0.22.0 --system --torch-backend=cu128
cd ..
