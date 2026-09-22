# 人員停留數系統

雙鏡頭人員停留數計算系統。程式依 Stable UID 的移動軌跡與跨線方向更新人員停留數，並保存偵測截圖與事件紀錄。

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

新攝影機可先用有人通行的影片自動估算主要動線、瓶頸位置與線長，再於預覽畫面確認：

```bash
python tools/calibrate_line.py --config config.json --camera top --auto-video calibration/top_people.mp4
```

若已人工確認進出方向，可加 `--save-auto` 直接儲存幾何結果。影像只能推測動線，無法自行知道哪一側在業務上代表「進入」，首次安裝仍需確認畫面上的 `+/-` 與 `direction` 對應。

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

箭頭起點到終點定義正負側；預設 `positive_to_negative` 為進入。辨識畫面會同時顯示計數用 Stable UID、模型原始 ID、移動軌跡與頭部／備援判斷點。

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
- 井底：`positive_to_negative` 為出，反向為進
- 計數狀態綁定 Stable UID，不直接綁模型原始 ID；原始 ID 跳動時會依預測位置與局部人體尺度做一對一交接
- 重疊人物的頭部偵測採一對一配對，不會把同一個頭點分給兩個人物框
- 追蹤點經多幀濾波，跨線前後需穩定並深入下游才確認；門檻隨人物框高度自適應，不再分 top/bottom 寫死像素跳動值
- 若人物越線後立即被遮擋，同一 ID、同一點來源且連續朝單一方向移動時，可用 raw 軌跡提前確認；單一跳點仍不計數
- 同一 Stable UID 未先發生反向事件，不會重複計算同方向事件
- 未完成跨線下游確認前折返不計數
- 多人可同時追蹤計數
- 不鎖定整體人流方向，可同時計算相反方向的不同人員
- 沒有追蹤 ID 時停止計數；不同 ID 間只在短時間、短距離且一對一時交接
- 人員停留數不允許低於 0
- 所有鏡頭的人形框先依人體框高度確認實際位移，再顯示與標註；`crossing.display_min_motion_box_ratio` 預設 `0.03`，設為 `0` 可停用
- 活躍追蹤逾一秒即失效，避免不同人員共用舊旅程；走廊旅程仍可依 `journey_reacquire_timeout_sec` 接回

## 影片回歸測試

下列命令會依序測試 `top_0831-0903.mp4`、`top_0910.mp4`、`bottom_0910.mp4`，並為每支來源輸出 H.264 辨識影片與事件 JSON：

```bash
python tools/video_regression.py
```
