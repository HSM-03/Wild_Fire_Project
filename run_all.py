"""전체 파이프라인을 순서대로 실행한다:  python run_all.py
(원자료는 raw/ 아래에 있어야 한다. README 참고)"""
import subprocess
import sys

for step in ["01_boundaries.py", "01b_landuse.py", "02_fires.py", "03_model.py", "04_build_dashboard.py"]:
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, f"pipeline/{step}"], check=True)
