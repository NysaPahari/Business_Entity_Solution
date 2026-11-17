"""Wait for host RAM before launching one isolated neural pilot."""
import json
import subprocess
import time
from pathlib import Path
import psutil


def main():
    out=Path('artifacts/accuracy/neural_probe'); out.mkdir(exist_ok=True)
    status=out/'launcher_status.json'
    while psutil.virtual_memory().available < 2*1024**3:
        status.write_text(json.dumps({'state':'waiting_for_2GiB_host_RAM'}))
        time.sleep(60)
    status.write_text(json.dumps({'state':'running'}))
    with Path('work_accuracy_test/neural_probe.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([str(Path('.neural-venv/Scripts/python.exe').resolve()),'-u','-m','src.neural_probe'],stdout=log,stderr=subprocess.STDOUT)
    status.write_text(json.dumps({'state':'complete' if result.returncode==0 else 'failed','returncode':result.returncode}))


if __name__=='__main__': main()
