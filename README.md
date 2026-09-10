# 人員停留數系統

雙鏡頭人員停留數計算系統。程式依每個追蹤 ID 的移動軌跡與跨線方向更新人員停留數，並保存偵測截圖與事件紀錄。

## 啟動

```bash
python main.py --config config.json
```

## 校正計數線

抓一張現場攝影機畫面，指定計數線起點與終點並存回 `config.json`：

```bash
python tools/calibrate_line.py --config config.json --camera top
python tools/calibrate_line.py --config config.json --camera bottom
```

也可以先用 ffmpeg 截原解析度圖片，再指定圖片校正：

```bash
mkdir -p calibration
ffmpeg -rtsp_transport tcp -i "rtsp://帳號:密碼@IP:PORT/路徑" -frames:v 1 -q:v 2 calibration/top.jpg
python tools/calibrate_line.py --config config.json --camera top --image calibration/top.jpg
```

操作：

- 滑鼠左鍵：依序指定起點、終點
- 滑鼠右鍵或 `U`：復原上一個點
- `R`：清空計數線
- `S`：儲存，必須正好 2 點
- `Q` 或 `Esc`：取消

箭頭起點到終點定義正負側；預設 `positive_to_negative` 為進入。辨識畫面會顯示計數線、追蹤 ID、移動軌跡與判斷點。判斷點預設在人框上方往下 15%，可用 `crossing.point_y_ratio` 調整。

## 安裝

```bash
./install.sh
```

安裝腳本會：

- 安裝系統套件與 Python 依賴
- 產生 `config.json`
- 建立 `img_log/personnel_count/` 與 `log/`
- 註冊並啟動 `systemctl --user` 服務：`personnel_count.service`

## 服務指令

```bash
systemctl --user status personnel_count.service
journalctl --user -u personnel_count.service -f
systemctl --user restart personnel_count.service
```

## 記錄

- 截圖：`img_log/personnel_count/`
- 系統 log：`log/personnel_count.log`

## 計數規則

- 井上：`positive_to_negative` 為進，反向為出
- 井底：`positive_to_negative` 為進，反向為出
- 同一追蹤 ID 完整穿越有限長度計數線才計數
- 中途折返不計數
- 多人可同時追蹤計數
- 沒有追蹤 ID 時停止計數，不使用鄰近座標猜測
- 人員停留數不允許低於 0
