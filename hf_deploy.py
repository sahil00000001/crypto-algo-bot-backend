"""Run this once to push all backend files to your HuggingFace Space."""
import os, sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import HfApi

TOKEN = input("Paste your HuggingFace token (from https://huggingface.co/settings/tokens): ").strip()
REPO  = "Sahilvashi123/crypto-algo-bot-backend"

api   = HfApi(token=TOKEN)
SKIP  = {"hf_deploy.py", "__pycache__", ".git", "trades.log", "trade_history.csv"}
SRC   = Path(__file__).parent

print(f"\nUploading to {REPO} ...")
for f in SRC.iterdir():
    if f.name in SKIP or f.name.startswith(".") or f.is_dir():
        continue
    print(f"  >> {f.name}")
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=f.name,
        repo_id=REPO,
        repo_type="space",
        token=TOKEN,
    )

print("\nDone! Go to https://huggingface.co/spaces/Sahilvashi123/crypto-algo-bot-backend")
print("It will build in ~2 minutes. Status turns green when ready.")
