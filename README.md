# 人員停留數系統

雙鏡頭人員停留數計算系統。程式偵測單人穿越畫面中的 A/B 區域，依方向更新人員停留數，並保存偵測截圖與事件紀錄。

## 啟動

```bash
python main.py --config config.json
```

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
- 事件 CSV：`log/personnel_count_events.csv`

## 計數規則

- 井上：`A_to_B` 為進，`B_to_A` 為出
- 井底：`A_to_B` 為進，`B_to_A` 為出
- 單人完整穿越 A/B 才計數
- 中途折返不計數
- 多人同框暫停計數
- 人員停留數不允許低於 0
