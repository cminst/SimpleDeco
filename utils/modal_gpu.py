import os
import secrets
from random import randint
import subprocess
import pathlib
import modal

image = (
    modal.Image.from_registry("qingy1337/graphml-base:cu128")
)

# Set environment variables within the Modal image.
image = image.env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_TOKEN": os.environ['HF_TOKEN'],
    "WANDB_API_KEY": os.environ['WANDB_API_KEY']
})

# Add system-level commands to the Modal image to update package lists and install essential tools.
image = image.run_commands(
    "apt-get update -y",
    "apt-get install git curl build-essential wget nano zip -y",
    "uv pip install jupyter jupyterlab huggingface_hub ipywidgets matplotlib nvitop --system"
)

# Download and install CUDA.
image = image.run_commands(
    "wget https://gist.githubusercontent.com/qingy1337/17e3dd68375388a171ea8b20919c61aa/raw/fc82544f52fa01e23a3f2da43263476da49381da/cuda128.sh",
    "bash cuda128.sh"
)

# Additional setup commands.
image = image.run_commands(
    "git config --global user.email \"qingy2019@outlook.com\"",
    "git config --global user.name \"Qingyun Li (from Modal)\"",
    f"echo \"https://qingy1337:{os.environ.get('GITHUB_PAT', '')}@github.com\" >> ~/.git-credentials",
    "git config --global credential.helper store",
    "uv pip install -U torchvision torchaudio torch==2.8.0 'numpy<2.3' --torch-backend=cu128 --no-build-isolation --system",
    "uv pip install accelerate trl==0.22.0 deepspeed orjson hf_transfer --system --torch-backend=cu128",
)

# ---------

app = modal.App(image=image, name="GPU Tasks")

@app.function(gpu="H100:1", timeout=86400)
def runwithgpu():
    token = 'modal'

    subprocess.run(["git", "clone", "https://github.com/cminst/SimpleDeco.git"], check=True)

    with modal.forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print('-'*50 + '\n' + f"{url}\n"+'-'*50)
        subprocess.run(
            [
                "uv",
                "run",
                "jupyter",
                "lab",
                "--no-browser", # Prevent JupyterLab from trying to open a browser automatically.
                "--allow-root", # Allow JupyterLab to be run as root user inside the container.
                "--ip=0.0.0.0", # Bind JupyterLab to all network interfaces, making it accessible externally.
                "--port=8888", # Specify the port for JupyterLab to listen on.
                "--LabApp.allow_origin='*'", # Allow requests from any origin (for easier access from different networks).
                "--LabApp.allow_remote_access=1", # Allow remote connections to JupyterLab.
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"}, # Set environment variables, including the authentication token and shell.
            # stderr=subprocess.PIPE,
            # stdout=subprocess.PIPE,
            check=True,
        )
        print("HERE")

@app.local_entrypoint()
def main():
    runwithgpu.remote()
