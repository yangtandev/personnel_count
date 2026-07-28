# 人員停留數系統

雙鏡頭人員停留數計算系統。程式偵測單人穿越畫面中的 A/B 區域，依方向更新人員停留數，並保存偵測截圖與事件紀錄。

## 啟動

```bash
python main.py --config config.json
```

## 校正 A/B 區

抓一張現場攝影機畫面，手動畫 A/B 區並存回 `config.json`：

```bash
python tools/calibrate_zones.py --config config.json --camera top
python tools/calibrate_zones.py --config config.json --camera bottom
```

也可以先用 ffmpeg 截原解析度圖片，再指定圖片校正：

```bash
mkdir -p calibration
ffmpeg -rtsp_transport tcp -i "rtsp://帳號:密碼@IP:PORT/路徑" -frames:v 1 -q:v 2 calibration/top.jpg
python tools/calibrate_zones.py --config config.json --camera top --image calibration/top.jpg
```

操作：

- 滑鼠左鍵：新增一個點
- `A` / `B`：切換目前畫的區域
- 滑鼠右鍵或 `U`：復原目前區域上一個點
- `R`：清空目前區域
- `S`：儲存，A/B 都至少 3 點
- `Q` 或 `Esc`：取消

儲存後，辨識畫面會顯示手動畫出的 A/B 區。判斷點在 A 區內是 A，在 B 區內是 B；兩區外不觸發 A/B 轉換。判斷點預設在人框上方往下 35%，可用 `zones.zone_point_y_ratio` 調整。

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

- 井上：`A_to_B` 為進，`B_to_A` 為出
- 井底：`A_to_B` 為進，`B_to_A` 為出
- A/B 區內人員完整穿越才計數
- 中途折返不計數
- A/B 區外人員忽略
- 多人可同時追蹤計數
- 人員停留數不允許低於 0
