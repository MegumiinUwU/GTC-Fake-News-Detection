#!/usr/bin/env python
import os
import sys
from pathlib import Path


def main():

    # Ensure project root is on sys.path so apps like ml_pipeline are importable
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'newsdetect.settings')
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()


