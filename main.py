import argparse
import sys

from personnel_count.config import DEFAULT_CONFIG_PATH
from personnel_count.qt_compat import configure_runtime_environment


def parse_args():
    parser = argparse.ArgumentParser(description="人員停留數主程式")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"設定檔路徑，預設 {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_runtime_environment()
    from personnel_count.app import run

    return run(args.config)


if __name__ == "__main__":
    sys.exit(main())
