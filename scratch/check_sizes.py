import subprocess
from pathlib import Path

files = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], text=True).strip().splitlines()

total_bytes = 0
max_file = ('', 0)
print(f"=== FILES TO BE COMMITTED ({len(files)} files) ===")
for f in files:
    p = Path(f)
    size = p.stat().st_size
    total_bytes += size
    if size > max_file[1]:
        max_file = (f, size)
    print(f"{f:<40} {size:>10,} bytes ({size/(1024*1024):.3f} MB)")

print("=" * 65)
bm = Path('models/best_model.joblib')
print(f"models/best_model.joblib size : {bm.stat().st_size:,} bytes ({bm.stat().st_size/(1024*1024):.2f} MB)")
print(f"Largest single file           : {max_file[0]} ({max_file[1]:,} bytes / {max_file[1]/(1024*1024):.2f} MB)")
print(f"Total size of all commit files: {total_bytes:,} bytes ({total_bytes/(1024*1024):.2f} MB)")
status = "PASSED (Well below GitHub 100 MB single file limit)" if max_file[1] < 100 * 1024 * 1024 else "FAILED"
print(f"GitHub 100MB limit status     : {status}")
