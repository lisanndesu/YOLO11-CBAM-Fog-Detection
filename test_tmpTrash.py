# get_env_eng_md.py
import platform
import subprocess
import sys

def gpu_info():
    """Return GPU name if NVIDIA present, else '-'"""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        return out.strip()
    except:
        return "-"

def pytorch_cuda():
    try:
        import torch
        return torch.__version__, torch.version.cuda or "-"
    except ImportError:
        return "-", "-"

def main():
    os_name   = platform.system() + " " + platform.release()
    cpu       = platform.processor() or platform.machine()
    # cross-platform RAM (only Linux shown)
    mem_gb    = "-" if platform.system() != "Linux" else \
                round(int(next(line.split()[1] for line in open("/proc/meminfo") if line.startswith("MemTotal:"))) / 1024**2)
    py_ver    = platform.python_version()
    pt_ver, cuda_ver = pytorch_cuda()
    gpu_name  = gpu_info()

    md = f"""| Environmental Parameter | Value |
|:--------------------------|:------|
| Operating System | {os_name} |
| Processor | {cpu} |
| System Memory | {mem_gb} GB |
| Python Version | {py_ver} |
| Deep Learning Framework | PyTorch {pt_ver} |
| CUDA Runtime | {cuda_ver} |
| Graphics Device | {gpu_name} |"""
    print(md)

if __name__ == "__main__":
    main()