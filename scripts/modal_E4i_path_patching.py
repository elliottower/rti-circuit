"""Modal wrapper for E4i_path_patching_proper.py.

Usage:
    modal run paper/modal_E4i_path_patching.py --detach
"""

import modal

app = modal.App("rti-path-patching-proper")

vol = modal.Volume.from_name("paper-path-patching-results", create_if_missing=True)

PART3_DIR = (
    "/Users/elliottower/Documents/GitHub/factorization-circuits"
    "/MIB/MIB-circuit-track/weight_circuit/experiments"
    "/v2_second_investigation/raw_experiments/v1_role_weight_analysis"
    "/part3_l9h3_investigation"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "transformer-lens==2.17.0",
        "transformers==4.51.3",
        "matplotlib==3.9.4",
    )
    .add_local_file(
        "paper/E4i_path_patching_proper.py",
        remote_path="/app/E4i_path_patching_proper.py",
    )
    .add_local_dir(
        PART3_DIR,
        remote_path="/app/a/b/c/d/e/part3",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": vol},
)
def run_path_patching():
    import os
    import sys
    import shutil
    from datetime import datetime

    os.environ["RTI_DIR"] = "/app/a/b/c/d/e/part3"
    os.environ["PP_OUT_DIR"] = "/results/E4i_path_patching_proper"
    sys.path.insert(0, "/app")

    print(f"[{datetime.now().isoformat()}] Starting proper path patching", flush=True)

    from E4i_path_patching_proper import run_all
    run_all(device="cuda")

    from modal import Volume
    vol = Volume.lookup("paper-path-patching-results")
    vol.commit()

    print(f"[{datetime.now().isoformat()}] Results committed to volume", flush=True)


@app.local_entrypoint()
def main():
    import time
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Launching path patching on Modal A10G")
    run_path_patching.remote()
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Done in {elapsed:.0f}s")
