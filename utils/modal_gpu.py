import os
import pathlib
import shutil
import subprocess
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
        "apt-get install -y git curl build-essential wget nano zip",
        "curl -fsSL https://tailscale.com/install.sh | sh",
        "uv pip install jupyter jupyterlab huggingface_hub ipywidgets matplotlib nvitop --system",
        "wget https://gist.githubusercontent.com/qingy1337/17e3dd68375388a171ea8b20919c61aa/raw/fc82544f52fa01e23a3f2da43263476da49381da/cuda128.sh",
        "bash cuda128.sh",
        'git config --global user.email "qingy2019@outlook.com"',
        'git config --global user.name "Qingyun Li (from Modal)"',
        'git config --global credential.helper store',
    )
    .run_commands("echo 'VERSION 1'")
    .run_commands(
        "uv pip install -U torchvision torchaudio torch==2.8.0 'numpy<2.3' vllm==0.10.2 --torch-backend=cu128 --no-build-isolation --system",
        "uv pip install accelerate trl==0.22.0 deepspeed orjson hf_transfer setuptools_scm --torch-backend=cu128 --system",
        "git clone https://github.com/cminst/SimpleDeco.git --recurse-submodules /tmp/SimpleDeco",
        "cd /tmp/SimpleDeco && bash utils/install_simpledeco_vllm_modal.sh",
    )
    .run_commands( # Additional apt installations after everything else
        "apt-get update && apt install openssh-client netcat-openbsd tmux sshpass -y"
    )
    .add_local_file("utils/tailscale-entrypoint.sh", "/root/entrypoint.sh", copy=True)
    .run_commands(
        "mkdir -p ~/.ssh",
        'echo "Host 100.84.104.59" >> ~/.ssh/config',
        'echo "    HostName 100.84.104.59" >> ~/.ssh/config',
        'echo "    User zli" >> ~/.ssh/config',
        'echo "    ProxyCommand nc -X 5 -x localhost:1080 %h %p" >> ~/.ssh/config',
        "chmod 600 ~/.ssh/config",
    )
    .run_commands("echo 'STARTING TAILSCALE'")
    .run_commands("chmod +x /root/entrypoint.sh")
    .entrypoint(["/root/entrypoint.sh"])
)

autodeco_volume = modal.Volume.from_name("autodeco", create_if_missing=True)
app = modal.App(image=image, name="AutoDeco Experiments")

@app.function(
    gpu="H200:1", # RTX-PRO-6000:1
    timeout=86400,
    volumes={"/root/SimpleDeco": autodeco_volume},
    secrets=[modal.Secret.from_name("tailscale-auth", required_keys=["TAILSCALE_AUTHKEY"])],
)
def runwithgpu():
    repo = pathlib.Path("/root/SimpleDeco")
    token = "modal"

    temp_clone_path = pathlib.Path("/tmp/SimpleDeco_temp_clone")

    if not (repo / ".git").exists():
        # Ensure temporary clone path is clean before cloning
        if temp_clone_path.exists():
            shutil.rmtree(temp_clone_path)

        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/cminst/SimpleDeco.git",
                "--recurse-submodules",
                str(temp_clone_path),
            ],
            check=True,
        )

        # Clear existing contents of the target volume mount point (`repo`)
        repo.mkdir(parents=True, exist_ok=True)
        if repo.exists():
            for item in repo.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        # `/root/SimpleDeco` is a mounted Modal volume, so moving the cloned
        # directory onto it fails. Copy the repo contents into the mount instead.
        shutil.copytree(temp_clone_path, repo, dirs_exist_ok=True, symlinks=True)
        shutil.rmtree(temp_clone_path)

        subprocess.run(
            [
                "hf",
                "download",
                "cminst/AutoDeco-R1-Distill-Qwen-7B-Merged",
                "--local-dir",
                "ckpt/AutoDeco-R1-Distill-Qwen-7B-merged",
            ],
            check=True,
            cwd=repo,
        )
        subprocess.run(
            [
                "hf",
                "download",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "--local-dir",
                "ckpt/DeepSeek-R1-Distill-Qwen-7B",
            ],
            check=True,
            cwd=repo,
        )

        autodeco_volume.commit()
    else:
        subprocess.run(
            ["git", "pull"],
            check=True,
            cwd=repo,
        )

    os.chdir("/root")

    # optional: confirm Tailscale is up
    subprocess.run(["tailscale", "status"], check=True)

    with modal.forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print("-" * 50)
        print(url)
        print("-" * 50)

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
