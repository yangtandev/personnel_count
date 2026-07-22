#!/bin/bash
set -e

echo "▶ 人員停留數系統安裝"
sudo -v
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CURRENT_USER="$USER"
PYTHON_CMD="python3"

cd "$PROJECT_DIR"

echo ""
read -p "請輸入井上 Camera RTSP 網址 [預設: rtsp://127.0.0.1/top]: " INPUT_TOP_CAMERA
TOP_CAMERA=${INPUT_TOP_CAMERA:-"rtsp://127.0.0.1/top"}

read -p "請輸入井底 Camera RTSP 網址 [預設: rtsp://127.0.0.1/bottom]: " INPUT_BOTTOM_CAMERA
BOTTOM_CAMERA=${INPUT_BOTTOM_CAMERA:-"rtsp://127.0.0.1/bottom"}

read -p "請輸入初始人員停留數 [預設: 0]: " INPUT_INITIAL_COUNT
INITIAL_COUNT=${INPUT_INITIAL_COUNT:-0}

echo ""
echo "▶ 設定確認"
echo "  - 專案路徑: $PROJECT_DIR"
echo "  - 執行身份: $CURRENT_USER"
echo "  - 井上攝影機: $TOP_CAMERA"
echo "  - 井底攝影機: $BOTTOM_CAMERA"
echo "  - 初始人員停留數: $INITIAL_COUNT"
echo ""
read -p "請按 Enter 鍵繼續安裝，或按 Ctrl+C 取消..."

echo "▶ [1/5] 安裝系統依賴套件..."
sudo apt-get update
sudo apt-get install -y build-essential curl wget git git-lfs python3 python3-pip python3-venv \
    ffmpeg libxcb-xinerama0 libxcb-xfixes0 libxcb-shape0 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    echo "▶ 系統 Python 低於 3.10，安裝 Python 3.10..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
    PYTHON_CMD="python3.10"
fi

echo "▶ 拉取模型檔案..."
git lfs install
git lfs pull

echo "▶ [2/5] 建立 Python venv 並安裝套件..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "▶ [3/5] 產生 config.json..."
python - <<EOF
import json
from pathlib import Path

config = {
    "camera": {
        "top": "$TOP_CAMERA",
        "bottom": "$BOTTOM_CAMERA",
        "camera_connect_timeout_sec": 5,
        "camera_reconnect_delay_sec": 2,
        "ffprobe_timeout": 5,
    },
    "model": {
        "path": "models/int8/best_cloth2_openvino_model",
        "person_class_id": 1,
        "min_conf": 0.5,
        "iou": 0.45,
        "inference_width": 960,
    },
    "counter": {
        "initial_count": int("$INITIAL_COUNT"),
        "min_person_area_ratio": 0.02,
        "lost_timeout_sec": 2.0,
        "event_cooldown_sec": 1.5,
    },
    "zones": {
        "zone_point_y_ratio": 0.35,
        "regions": {},
    },
    "direction": {
        "top": {"A_to_B": "enter", "B_to_A": "exit"},
        "bottom": {"B_to_A": "exit", "A_to_B": "enter"},
    },
    "storage": {
        "img_log_dir": "img_log/personnel_count",
        "log_dir": "log",
    },
    "ui": {
        "fullscreen": True,
        "window_title": "人員停留數",
        "camera_names": {
            "top": "井上",
            "bottom": "井底",
        },
    },
}

Path("img_log/personnel_count").mkdir(parents=True, exist_ok=True)
Path("log").mkdir(parents=True, exist_ok=True)
Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
Path(".cache/yolo").mkdir(parents=True, exist_ok=True)
with open("config.json", "w", encoding="utf-8") as fh:
    json.dump(config, fh, ensure_ascii=False, indent=2)
EOF

echo "▶ [4/5] 註冊 systemd user service..."
USER_SERVICE_DIR="$HOME/.config/systemd/user"
mkdir -p "$USER_SERVICE_DIR"
SERVICE_FILE_PATH="$USER_SERVICE_DIR/personnel_count.service"
USER_ID=$(id -u "$CURRENT_USER")

tee "$SERVICE_FILE_PATH" > /dev/null <<EOF
[Unit]
Description=Personnel Count Service
After=network.target graphical-session.target

[Service]
Type=simple
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=/run/user/$USER_ID
Environment=XAUTHORITY=$HOME/.Xauthority
Environment=PYTHONUNBUFFERED=1
Environment=MPLCONFIGDIR=$PROJECT_DIR/.cache/matplotlib
Environment=YOLO_CONFIG_DIR=$PROJECT_DIR/.cache/yolo
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python -u $PROJECT_DIR/main.py --config $PROJECT_DIR/config.json
Restart=always
RestartSec=10

[Install]
WantedBy=graphical-session.target
EOF

systemctl --user daemon-reload
systemctl --user enable personnel_count.service
systemctl --user restart personnel_count.service || true

echo "▶ [5/5] 安裝完成"
echo ""
echo "狀態：systemctl --user status personnel_count.service"
echo "日誌：journalctl --user -u personnel_count.service -f"
echo "截圖：$PROJECT_DIR/img_log/personnel_count"
