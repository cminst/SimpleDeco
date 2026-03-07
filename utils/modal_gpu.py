import os
import subprocess
import pathlib
import modal

image = (
    modal.Image.from_registry("qingy1337/graphml-base:cu128")
    .workdir("/root")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
    })
    .run_commands(
        "apt-get update -y",
        "apt-get install git curl build-essential wget nano zip -y",
        "uv pip install jupyter jupyterlab huggingface_hub ipywidgets matplotlib nvitop --system",
        "wget https://gist.githubusercontent.com/qingy1337/17e3dd68375388a171ea8b20919c61aa/raw/fc82544f52fa01e23a3f2da43263476da49381da/cuda128.sh",
        "bash cuda128.sh",
        'git config --global user.email "qingy2019@outlook.com"',
        'git config --global user.name "Qingyun Li (from Modal)"',
        f'echo "https://qingy1337:{os.environ.get("GITHUB_PAT", "")}@github.com" >> ~/.git-credentials',
        "git config --global credential.helper store"
    )
    .run_commands(
        "uv pip install -U torchvision torchaudio torch==2.8.0 'numpy<2.3' vllm==0.10.2 --torch-backend=cu128 --no-build-isolation --system",
        "uv pip install accelerate trl==0.22.0 deepspeed orjson hf_transfer setuptools_scm --torch-backend=cu128 --system",

        "git clone https://github.com/cminst/SimpleDeco.git --recurse-submodules /tmp/SimpleDeco",
        "cd /tmp/SimpleDeco && bash utils/install_simpledeco_vllm_modal.sh",
    )
)

autodeco_volume = modal.Volume.from_name("autodeco", create_if_missing=True)

app = modal.App(image=image, name="AutoDeco Experiments")


@app.function(
    gpu="A100-40GB:1",
    timeout=86400,
    volumes={"/root/SimpleDeco": autodeco_volume},
)
def runwithgpu():
    repo = pathlib.Path("/root/SimpleDeco")
    token = "modal"

    # First run only: populate the volume-backed repo
    if not repo.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/cminst/SimpleDeco.git",
                "--recurse-submodules",
                str(repo),
            ],
            check=True,
        )
        autodeco_volume.commit()  # optional, but nice for the initial seed

    os.chdir("/root")

    with modal.forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print("-" * 50 + "\n" + f"{url}\n" + "-" * 50)

        subprocess.run(
            [
                "uv",
                "run",
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--LabApp.allow_origin=*",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            check=True,
        )


@app.local_entrypoint()
def main():
    runwithgpu.remote()
