STATUS_TEXT = {
    "starting": "啟動中",
    "camera_waiting": "等待攝影機畫面",
    "detector_error": "偵測異常",
    "waiting": "等待人員通過",
    "tracking": "追蹤人員移動中",
    "incomplete_path": "路徑未完成，未計數",
    "cooldown": "已計數，等待人員離開",
    "unknown_direction": "方向不明，未計數",
    "counted_enter": "已計入進入",
    "counted_exit": "已計入離開",
    "near_line": "人員接近計數線",
    "crossing_rejected": "未穿越有效線段，不計數",
    "tracking_unavailable": "追蹤 ID 無法使用，停止計數",
    "line_not_configured": "尚未設定計數線",
    "corridor_not_configured": "尚未設定通行區域",
    "corridor_pending": "確認人員所在區域",
    "corridor_between_zones": "人員位於通道邊界",
    "corridor_in_transit": "人員通過計數通道",
    "corridor_transit_unarmed": "通道內新出現人員，不計數",
    "corridor_turnback": "人員折返，未計數",
    "corridor_baseline": "啟動時既有人員，不計數",
    "corridor_handoff_pending": "確認換軌後的人員身分",
    "flow_reverse_ignored": "反向事件已由方向鎖忽略",
    "stabilizing": "確認軌跡起始側",
    "rearming": "等待軌跡重新定位",
    "crossing_pending": "確認跨線方向",
    "point_jump": "追蹤點跳動，重新定位",
}


def user_status_text(status):
    return STATUS_TEXT.get(status, "系統運作中")
