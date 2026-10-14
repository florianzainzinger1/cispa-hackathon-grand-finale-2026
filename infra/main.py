"""Smoke-test entry point for the JURECA pipeline.

Verifies that the Slurm job runs, the venv is active, and a GPU is visible.
Replace this with the real FL_Data_Reconstruction task code.
"""

import platform
import socket
import sys


def main() -> None:
    print("=== JURECA pipeline smoke test ===")
    print(f"Hostname:  {socket.gethostname()}")
    print(f"Python:    {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform:  {platform.platform()}")

    try:
        import torch

        print(f"PyTorch:   {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU:       {torch.cuda.get_device_name(0)}")
            x = torch.rand(1024, 1024, device="cuda")
            print(f"GPU matmul OK, result sum: {(x @ x).sum().item():.2f}")
    except ImportError:
        print("PyTorch not installed in this environment.")

    print("=== Smoke test finished ===")


if __name__ == "__main__":
    main()
