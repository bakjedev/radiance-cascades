import argparse
import re
import json
import hashlib
import os
import sys
import shutil
import subprocess
from pathlib import Path

SHADER_ENTRY = re.compile(rb'\[\s*shader\s*\(')
DEP_TOKEN = re.compile(r'(?:\\ |\S)+')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, type=Path)
    parser.add_argument('--dst', required=True, type=Path)
    parser.add_argument('--slangc', required=True)
    parser.add_argument('--stamp', required=True, type=Path)
    parser.add_argument('--slangc-args', nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def parse_depfile(path: Path):
    text = path.read_text(errors='replace').replace('\\\r\n', ' ').replace('\\\n', ' ')
    deps = []
    for line in text.splitlines():
        _, sep, rhs = line.partition(': ')
        if not sep:
            continue
        for token in DEP_TOKEN.findall(rhs):
            if not token.endswith(':'):
                deps.append(token.replace('\\ ', ' '))
    return deps


def main():
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    shader_src = src / 'shaders'
    cache = dst / '.cache'
    cache.mkdir(parents=True, exist_ok=True)

    flags_file = cache / 'slang_flags'
    flags_key = hashlib.sha1(json.dumps([args.slangc, args.slangc_args]).encode()).hexdigest()
    try:
        flags_changed = flags_file.read_text() != flags_key
    except OSError:
        flags_changed = True

    copies, shaders, all_slang = [], [], []
    for path in sorted(src.rglob('*')):
        rel = path.relative_to(src)
        if any(part.startswith('.') for part in rel.parts) or not path.is_file():
            continue
        if shader_src in path.parents:
            if path.suffix == '.slang':
                all_slang.append(str(path))
                if SHADER_ENTRY.search(path.read_bytes()):
                    shaders.append((path, rel.with_suffix('.spv')))
            continue
        copies.append((path, rel))

    copy_count = 0
    for path, rel in copies:
        copy_dst = dst / rel
        stats = path.stat()
        try:
            dst_stats = copy_dst.stat()
            if dst_stats.st_size == stats.st_size and dst_stats.st_mtime_ns == stats.st_mtime_ns:
                continue
        except FileNotFoundError:
            pass
        copy_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy_dst)
        copy_count += 1

    def depfile_for(out_rel: Path):
        return cache / (out_rel.as_posix().replace('/', '__') + '.d')

    def is_outdated(s: Path, out_rel: Path):
        out = dst / out_rel
        if flags_changed or not out.exists():
            return True

        depfile = depfile_for(out_rel)
        deps = parse_depfile(depfile) if depfile.exists() else all_slang
        time = out.stat().st_mtime_ns
        for dep in {str(s), *deps}:
            try:
                if Path(dep).stat().st_mtime_ns > time:
                    return True
            except FileNotFoundError:
                return True
        return False

    compile_count, fails = 0, []
    for s, out_rel in shaders:
        if not is_outdated(s, out_rel):
            continue
        out = dst / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + '.temp.spv')
        depfile = depfile_for(out_rel)
        depfile.unlink(missing_ok=True)
        cmd = [args.slangc, str(s), '-I', str(shader_src), *args.slangc_args, '-o', str(tmp), '-depfile', str(depfile)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        key = out_rel.as_posix()
        if run.returncode != 0:
            tmp.unlink(missing_ok=True)
            out.unlink(missing_ok=True)
            fails.append(key)
            sys.stderr.write(f'[assets] FAILED {key}:\n{run.stderr}\n')
            continue
        os.replace(tmp, out)
        compile_count += 1
        if run.stderr.strip():
            sys.stderr.write(f'[assets] {key}:\n{run.stderr}\n')

    flags_file.write_text(flags_key)

    if copy_count or compile_count or fails:
        print(f'[assets] copied {copy_count}, compiled {compile_count}, failed {len(fails)}')

    if fails:
        return 1
    args.stamp.touch()
    return 0


if __name__ == '__main__':
    sys.exit(main())
