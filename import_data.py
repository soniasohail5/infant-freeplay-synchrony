import subprocess

print("Syncing dataset...")
subprocess.run(["bash", "sync_data.sh"], check=True)

print("Dataset import is complete.")
