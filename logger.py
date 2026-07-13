from datetime import datetime

LOG_FILE = "detect_log.txt"


def write_log(source, obj_name, conf, status):
    """
    source: image / video / camera
    obj_name: person/car...
    conf: confidence
    status: detected / alert
    """
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_text = f"[{time_now}] 来源:{source} | 目标:{obj_name} | 置信度:{conf:.2f} | 状态:{status}\n"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_text)