import argparse
import sys

from config.loader import DEFAULT_CONFIG_PATH
from ui.qt_compat import configure_runtime_environment


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
    from app.main import run

    return run(args.config)


if __name__ == "__main__":
    sys.exit(main())
