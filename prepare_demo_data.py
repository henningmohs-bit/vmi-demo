"""
One-off utility: trims the huge SIMION 'result' trajectory files down to a
small number of ions so they can be committed to git and preloaded on Render
(GitHub blocks files >100MB, Render free tier has only 512MB RAM).
Run once locally: python prepare_demo_data.py
"""

import re
import shutil
from pathlib import Path

MAX_IONS = 15
ION_START_RE = re.compile(r'^Ion\((\d+)\)\s+Event\(')

SOURCES = [
    (Path('Previous design'), Path('demo_data/previous_design')),
    (Path('Slot-free design'), Path('demo_data/slotfree_design')),
]


def trim_result_file(src: Path, dst: Path, max_ion: int):
    keep = False
    with open(src, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(dst, 'w', encoding='utf-8') as fout:
        for line in fin:
            m = ION_START_RE.match(line)
            if m:
                keep = int(m.group(1)) <= max_ion
            if keep:
                fout.write(line)


def main():
    for src_dir, dst_dir in SOURCES:
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_result = src_dir / 'result'
        dst_result = dst_dir / 'result'
        print(f"Trimming {src_result} -> {dst_result} (first {MAX_IONS} ions)")
        trim_result_file(src_result, dst_result, MAX_IONS)
        print(f"  {src_result.stat().st_size/1e6:.1f} MB -> {dst_result.stat().st_size/1e6:.2f} MB")

        for extra in ['charges.txt', *[f'electrode{i}.stl' for i in range(1, 10)]]:
            src_extra = src_dir / extra
            if src_extra.exists():
                shutil.copy2(src_extra, dst_dir / extra)

    print("Done. Demo data is in the 'demo_data' folder.")


if __name__ == '__main__':
    main()
