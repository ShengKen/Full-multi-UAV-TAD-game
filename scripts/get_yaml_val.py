#!/usr/bin/env python3


import sys

import yaml


def get_by_path(d, path: str):
    cur = d
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("用法: get_yaml_val.py <yaml文件> <点路径> [点路径 ...]", file=sys.stderr)
        sys.exit(2)
    yaml_file, paths = args[0], args[1:]
    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    parts = []
    for path in paths:
        try:
            parts.append((path, get_by_path(data, path)))
        except (KeyError, IndexError, ValueError) as e:
            print(f"路径 '{path}' 不存在: {e}", file=sys.stderr)
            sys.exit(1)

    if len(parts) == 1:
        print(parts[0][1])
    else:
        print(" ".join(f"{p}={v}" for p, v in parts))


if __name__ == "__main__":
    main()
