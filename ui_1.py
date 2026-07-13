import sys
import cv2
import os
import traceback
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint

from detector_image import detect_image
from detector_video import detect_video


# ============================================================
#  全局样式 — 简洁白色风 + 大字体
# ============================================================
LIGHT_STYLE = """
QMainWindow {
    background: #f8fafc;
}

/* ---------- 左侧导航 ---------- */
#navPanel {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
#navTitle {
    color: #1e293b;
    font-size: 26px;
    font-weight: bold;
    padding: 24px;
}
#navBtn {
    background: transparent;
    color: #64748b;
    border: none;
    border-left: 5px solid transparent;
    padding: 20px 28px;
    font-size: 20px;
    font-weight: 600;
    text-align: left;
}
#navBtn:hover {
    background: #f1f5f9;
    color: #334155;
}
#navBtn:checked, #navBtn:active {
    background: #eff6ff;
    color: #2563eb;
    border-left: 5px solid #2563eb;
}
#navExit {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 16px;
    font-size: 18px;
    font-weight: bold;
}
#navExit:hover {
    background: #fee2e2;
}

/* ---------- 内容区域 ---------- */
#contentPanel {
    background: transparent;
}
#card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
}
#cardTitle {
    color: #64748b;
    font-size: 16px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}
#cardValue {
    color: #1e293b;
    font-size: 38px;
    font-weight: bold;
}
#subText {
    color: #64748b;
    font-size: 18px;
}

/* ---------- 主标题 ---------- */
#bigTitle {
    color: #0f172a;
    font-size: 40px;
    font-weight: bold;
}
#subTitle {
    color: #64748b;
    font-size: 20px;
}

/* ---------- 按钮 ---------- */
#actionBtn {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 16px 36px;
    font-size: 18px;
    font-weight: bold;
}
#actionBtn:hover {
    background: #1d4ed8;
}
#actionBtn:pressed {
    background: #1e40af;
}
#secondaryBtn {
    background: #ffffff;
    color: #334155;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    padding: 16px 36px;
    font-size: 18px;
    font-weight: bold;
}
#secondaryBtn:hover {
    background: #f8fafc;
}

/* ---------- 图像显示区 ---------- */
#displayArea {
    background: #f1f5f9;
    border: 2px dashed #cbd5e1;
    border-radius: 20px;
}

/* ---------- 风险指示器 ---------- */
#riskPanel {
    border-radius: 16px;
    padding: 20px;
}
#riskSAFE {
    background: #f0fdf4;
    color: #16a34a;
    border: 1px solid #bbf7d0;
}
#riskMEDIUM {
    background: #fff7ed;
    color: #ea580c;
    border: 1px solid #fed7aa;
}
#riskHIGH {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}

/* ---------- 日志框 ---------- */
#logBox {
    background: #f8fafc;
    color: #475569;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 15px;
}

/* ---------- 表格 ---------- */
QTableWidget {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    color: #334155;
    gridline-color: #f1f5f9;
    font-size: 16px;
}
QTableWidget::item {
    padding: 12px;
    border-bottom: 1px solid #f1f5f9;
}
QTableWidget::item:selected {
    background: #dbeafe;
    color: #1d4ed8;
}
QHeaderView::section {
    background: #f8fafc;
    color: #64748b;
    padding: 14px;
    border: none;
    border-bottom: 2px solid #e2e8f0;
    font-weight: bold;
    font-size: 16px;
}

/* ---------- 进度条 ---------- */
QProgressBar {
    background: #e2e8f0;
    border-radius: 8px;
    height: 10px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: #2563eb;
    border-radius: 8px;
}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {
    background: #f1f5f9;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}
"""


# ============================================================
#  高级主界面 — 简洁白色版
# ============================================================
class AdvancedUI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("森林盗伐智能检测系统 ")
        self.resize(1920, 1080)
        self.setStyleSheet(LIGHT_STYLE)

        # ---------- 状态 ----------
        self.video_generator = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.paused = False
        self.total_frames = 0
        self.detected_person = 0
        self.detected_axe = 0
        self.detected_chainsaw = 0
        self.high_risk_count = 0
        self.medium_risk_count = 0
        self._last_risk = None
        self._video_writer = None
        self._video_save_path = None

        # ---------- 中心控件 ----------
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------- 右侧内容 ----------
        self.stack = QStackedWidget()
        self.stack.setObjectName("contentPanel")

        # ---------- 左侧导航 ----------
        self.nav_panel = self._build_nav()
        main_layout.addWidget(self.nav_panel)
        main_layout.addWidget(self.stack)

        # ---------- 页面 ----------
        self.page_home = self._build_home()
        self.page_image = self._build_image()
        self.page_video = self._build_video()

        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_image)
        self.stack.addWidget(self.page_video)

        # ---------- 时钟 ----------
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        # ---------- 入场动画 ----------
        self._animate_entry()

    # ============================ 导航栏 ============================
    def _build_nav(self):
        panel = QWidget()
        panel.setObjectName("navPanel")
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 24, 0, 24)
        layout.setSpacing(6)

        # 标题区
        title = QLabel("森林监控中心")
        title.setObjectName("navTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(line)
        layout.addSpacing(24)

        # 导航按钮组
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        self.btn_home = self._nav_button("系统总览")
        self.btn_img = self._nav_button("图片检测")
        self.btn_video = self._nav_button("视频检测")

        self.nav_group.addButton(self.btn_home, 0)
        self.nav_group.addButton(self.btn_img, 1)
        self.nav_group.addButton(self.btn_video, 2)

        self.nav_group.buttonClicked[int].connect(self.stack.setCurrentIndex)
        self.btn_home.setChecked(True)

        layout.addWidget(self.btn_home)
        layout.addWidget(self.btn_img)
        layout.addWidget(self.btn_video)
        layout.addStretch()

        # 状态信息
        self.status_label = QLabel("模型状态: 就绪\nYOLOv8 已加载")
        self.status_label.setObjectName("subText")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addSpacing(24)

        # 退出按钮
        btn_exit = QPushButton("退出系统")
        btn_exit.setObjectName("navExit")
        btn_exit.setCursor(Qt.PointingHandCursor)
        btn_exit.clicked.connect(self.close)
        layout.addWidget(btn_exit)

        return panel

    def _nav_button(self, text):
        btn = QPushButton(f"  {text}")
        btn.setObjectName("navBtn")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(64)
        return btn

    # ============================ 首页仪表盘 ============================
    def _build_home(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(48, 36, 48, 36)
        layout.setSpacing(28)

        # 顶部标题 + 时间
        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("森林盗伐智能检测系统")
        title.setObjectName("bigTitle")
        sub = QLabel("YOLOv8 深度目标检测 · 实时风险预警 · 智能分析平台")
        sub.setObjectName("subTitle")
        title_block.addWidget(title)
        title_block.addWidget(sub)
        header.addLayout(title_block)
        header.addStretch()
        self.clock_label = QLabel()
        self.clock_label.setObjectName("cardValue")
        self.clock_label.setStyleSheet("color: #2563eb; font-size: 32px;")
        header.addWidget(self.clock_label)
        layout.addLayout(header)

        # 统计卡片区
        cards = QHBoxLayout()
        cards.setSpacing(24)
        self.stat_total = self._stat_card("总检测次数", "0", "#2563eb")
        self.stat_safe = self._stat_card("安全状态", "0", "#16a34a")
        self.stat_warning = self._stat_card("中风险预警", "0", "#ea580c")
        self.stat_danger = self._stat_card("高风险告警", "0", "#dc2626")
        cards.addWidget(self.stat_total)
        cards.addWidget(self.stat_safe)
        cards.addWidget(self.stat_warning)
        cards.addWidget(self.stat_danger)
        layout.addLayout(cards)

        # 中部内容
        mid = QHBoxLayout()
        mid.setSpacing(24)

        # 左侧：模型信息
        info_card = self._card_widget()
        info_layout = QVBoxLayout(info_card)
        info_title = QLabel("系统信息")
        info_title.setObjectName("cardTitle")
        info_layout.addWidget(info_title)

        info_items = [
            ("检测模型", "yolov8s迁移训练模型"),
            ("检测类别", "person, axe, chainsaw"),
            ("置信度阈值", "0.10"),
            ("IOU 阈值", "0.20"),
            ("输入尺寸", "960 x 960"),
            ("运行平台", "CUDA GPU"),
        ]
        for label, value in info_items:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setObjectName("subText")
            lbl.setFixedWidth(140)
            val = QLabel(value)
            val.setStyleSheet("color: #334155; font-size: 17px;")
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            info_layout.addLayout(row)
        info_layout.addStretch()
        mid.addWidget(info_card, 1)

        # 右侧：风险规则
        rule_card = self._card_widget()
        rule_layout = QVBoxLayout(rule_card)
        rule_title = QLabel("风险判定规则")
        rule_title.setObjectName("cardTitle")
        rule_layout.addWidget(rule_title)

        rules = [
            ("SAFE", "未检测到人员或仅检测到人员", "#16a34a"),
            ("MEDIUM", "人员 + 斧头（潜在盗伐）", "#ea580c"),
            ("HIGH", "人员 + 电锯（高度危险）", "#dc2626"),
        ]
        for level, desc, color in rules:
            row = QHBoxLayout()
            lvl = QLabel(level)
            lvl.setStyleSheet(
                f"color: {color}; font-weight: bold; font-size: 17px; padding: 6px 16px; border-radius: 8px; background: {color}15;"
            )
            d = QLabel(desc)
            d.setStyleSheet("color: #475569; font-size: 17px;")
            row.addWidget(lvl)
            row.addWidget(d)
            row.addStretch()
            rule_layout.addLayout(row)
        rule_layout.addStretch()
        mid.addWidget(rule_card, 1)

        layout.addLayout(mid)

        # 底部：操作入口
        action_card = self._card_widget()
        action_layout = QHBoxLayout(action_card)
        action_layout.setSpacing(36)
        action_layout.addStretch()

        btn_img = QPushButton("开始图片检测")
        btn_img.setObjectName("actionBtn")
        btn_img.setCursor(Qt.PointingHandCursor)
        btn_img.setFixedSize(260, 68)
        btn_img.clicked.connect(lambda: self._switch_page(1))

        btn_video = QPushButton("开始视频检测")
        btn_video.setObjectName("actionBtn")
        btn_video.setCursor(Qt.PointingHandCursor)
        btn_video.setFixedSize(260, 68)
        btn_video.clicked.connect(lambda: self._switch_page(2))

        action_layout.addWidget(btn_img)
        action_layout.addWidget(btn_video)
        action_layout.addStretch()
        layout.addWidget(action_card)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _stat_card(self, title, value, color):
        card = self._card_widget()
        layout = QVBoxLayout(card)
        layout.setSpacing(10)

        t = QLabel(title)
        t.setObjectName("cardTitle")

        v = QLabel(value)
        v.setObjectName("cardValue")
        v.setStyleSheet(f"color: {color};")
        if title == "总检测次数":
            self.val_total = v
        elif title == "安全状态":
            self.val_safe = v
        elif title == "中风险预警":
            self.val_warning = v
        elif title == "高风险告警":
            self.val_danger = v

        layout.addWidget(t)
        layout.addWidget(v)
        return card

    def _card_widget(self):
        w = QWidget()
        w.setObjectName("card")
        w.setStyleSheet("""
            QWidget#card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 20px;
            }
        """)
        return w

    def _update_clock(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clock_label.setText(now)

    def _switch_page(self, index):
        self.stack.setCurrentIndex(index)
        btn = self.nav_group.button(index)
        if btn:
            btn.setChecked(True)

    # ============================ 图片检测页 ============================
    def _build_image(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(24)

        # 标题
        header = QHBoxLayout()
        title = QLabel("图片检测")
        title.setObjectName("bigTitle")
        header.addWidget(title)
        header.addStretch()
        self.img_time_label = QLabel()
        self.img_time_label.setObjectName("subText")
        header.addWidget(self.img_time_label)
        layout.addLayout(header)

        # 主体
        body = QHBoxLayout()
        body.setSpacing(24)

        # 左侧：图像显示
        left = QVBoxLayout()
        self.img_display = QLabel()
        self.img_display.setObjectName("displayArea")
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.setMinimumSize(900, 700)
        self.img_display.setText("点击右侧按钮选择图片进行检测")
        self.img_display.setStyleSheet("color: #94a3b8; font-size: 18px;")
        left.addWidget(self.img_display)
        body.addLayout(left, 3)

        # 右侧：控制 + 结果
        right = QVBoxLayout()
        right.setSpacing(20)

        # 操作按钮
        btn_select = QPushButton("选择图片")
        btn_select.setObjectName("actionBtn")
        btn_select.setCursor(Qt.PointingHandCursor)
        btn_select.setFixedHeight(56)
        btn_select.clicked.connect(self._run_image)
        right.addWidget(btn_select)

        # 结果卡片
        result_card = self._card_widget()
        result_layout = QVBoxLayout(result_card)
        result_title = QLabel("检测结果")
        result_title.setObjectName("cardTitle")
        result_layout.addWidget(result_title)

        self.img_result_table = QTableWidget()
        self.img_result_table.setColumnCount(3)
        self.img_result_table.setHorizontalHeaderLabels(["类别", "置信度", "边界框"])
        self.img_result_table.horizontalHeader().setStretchLastSection(True)
        self.img_result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.img_result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.img_result_table.setMinimumHeight(220)
        result_layout.addWidget(self.img_result_table)

        # 风险状态
        self.img_risk_panel = QLabel("风险状态: 等待检测")
        self.img_risk_panel.setObjectName("riskPanel")
        self.img_risk_panel.setStyleSheet("""
            QLabel {
                background: #f8fafc;
                color: #64748b;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 18px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self.img_risk_panel.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.img_risk_panel)

        right.addWidget(result_card)

        # 日志
        log_card = self._card_widget()
        log_layout = QVBoxLayout(log_card)
        log_title = QLabel("运行日志")
        log_title.setObjectName("cardTitle")
        log_layout.addWidget(log_title)
        self.img_log = QTextEdit()
        self.img_log.setObjectName("logBox")
        self.img_log.setReadOnly(True)
        self.img_log.setMaximumHeight(180)
        log_layout.addWidget(self.img_log)
        right.addWidget(log_card)

        body.addLayout(right, 1)
        layout.addLayout(body)
        return page

    def _run_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file:
            return

        self.img_time_label.setText(f"检测时间: {datetime.now().strftime('%H:%M:%S')}")
        self._log(self.img_log, f"开始检测: {os.path.basename(file)}")

        try:
            img, detections = detect_image(file, save=True, for_ui=True)
        except Exception as e:
            self._log(self.img_log, f"检测出错: {e}")
            return

        if img is None:
            self._log(self.img_log, "读取图片失败")
            return

        self.display_image(img, self.img_display)

        # 填充结果表格
        self.img_result_table.setRowCount(0)
        person = axe = chainsaw = 0
        for det in detections:
            row = self.img_result_table.rowCount()
            self.img_result_table.insertRow(row)
            self.img_result_table.setItem(row, 0, QTableWidgetItem(det["label"]))
            self.img_result_table.setItem(row, 1, QTableWidgetItem(f"{det['score']:.3f}"))
            self.img_result_table.setItem(row, 2, QTableWidgetItem(str(det["box"])))

            if det["label"] == "person":
                person += 1
            elif det["label"] == "axe":
                axe += 1
            elif det["label"] == "chainsaw":
                chainsaw += 1

        # 风险判断
        risk, risk_style = self._risk_style(person, axe, chainsaw)
        self.img_risk_panel.setText(f"风险状态: {risk}")
        self.img_risk_panel.setStyleSheet(risk_style)

        self._log(self.img_log, f"检测完成: {len(detections)} 个目标 | {risk}")
        self._update_stats(risk)

    # ============================ 视频检测页 ============================
    def _build_video(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(24)

        # 标题
        header = QHBoxLayout()
        title = QLabel("视频检测")
        title.setObjectName("bigTitle")
        header.addWidget(title)
        header.addStretch()
        self.vid_time_label = QLabel()
        self.vid_time_label.setObjectName("subText")
        header.addWidget(self.vid_time_label)
        layout.addLayout(header)

        # 主体
        body = QHBoxLayout()
        body.setSpacing(24)

        # 左侧：视频显示
        left = QVBoxLayout()
        self.video_display = QLabel()
        self.video_display.setObjectName("displayArea")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(960, 720)
        self.video_display.setText("选择视频开始检测")
        self.video_display.setStyleSheet("color: #94a3b8; font-size: 18px;")
        left.addWidget(self.video_display)

        # 底部控制条
        controls = QHBoxLayout()
        controls.setSpacing(14)

        self.btn_select_vid = QPushButton("选择视频")
        self.btn_select_vid.setObjectName("actionBtn")
        self.btn_select_vid.setCursor(Qt.PointingHandCursor)
        self.btn_select_vid.setFixedHeight(52)
        self.btn_select_vid.clicked.connect(self._run_video)

        self.btn_pause_vid = QPushButton("暂停")
        self.btn_pause_vid.setObjectName("secondaryBtn")
        self.btn_pause_vid.setCursor(Qt.PointingHandCursor)
        self.btn_pause_vid.setFixedHeight(52)
        self.btn_pause_vid.setEnabled(False)
        self.btn_pause_vid.clicked.connect(self._toggle_pause)

        self.btn_stop_vid = QPushButton("停止")
        self.btn_stop_vid.setObjectName("secondaryBtn")
        self.btn_stop_vid.setCursor(Qt.PointingHandCursor)
        self.btn_stop_vid.setFixedHeight(52)
        self.btn_stop_vid.setEnabled(False)
        self.btn_stop_vid.clicked.connect(self._stop_video)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setValue(0)

        controls.addWidget(self.btn_select_vid)
        controls.addWidget(self.btn_pause_vid)
        controls.addWidget(self.btn_stop_vid)
        controls.addWidget(self.progress_bar, 1)
        left.addLayout(controls)
        body.addLayout(left, 3)

        # 右侧：实时数据
        right = QVBoxLayout()
        right.setSpacing(20)

        # 风险大字
        self.vid_risk_big = QLabel("SAFE")
        self.vid_risk_big.setAlignment(Qt.AlignCenter)
        self.vid_risk_big.setStyleSheet("""
            QLabel {
                background: #f0fdf4;
                color: #16a34a;
                border: 2px solid #bbf7d0;
                border-radius: 24px;
                padding: 36px;
                font-size: 52px;
                font-weight: bold;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
        """)
        right.addWidget(self.vid_risk_big)

        # 实时统计卡片
        stats = QHBoxLayout()
        self.vid_stat_person = self._mini_card("人员", "0", "#2563eb")
        self.vid_stat_axe = self._mini_card("斧头", "0", "#ea580c")
        self.vid_stat_chainsaw = self._mini_card("电锯", "0", "#dc2626")
        stats.addWidget(self.vid_stat_person)
        stats.addWidget(self.vid_stat_axe)
        stats.addWidget(self.vid_stat_chainsaw)
        right.addLayout(stats)

        # 帧信息
        info_card = self._card_widget()
        info_layout = QVBoxLayout(info_card)
        info_title = QLabel("实时信息")
        info_title.setObjectName("cardTitle")
        info_layout.addWidget(info_title)
        self.vid_frame_info = QLabel("帧: 0 | 状态: 待机")
        self.vid_frame_info.setStyleSheet("color: #475569; font-size: 17px;")
        info_layout.addWidget(self.vid_frame_info)
        right.addWidget(info_card)

        # 日志
        log_card = self._card_widget()
        log_layout = QVBoxLayout(log_card)
        log_title = QLabel("检测日志")
        log_title.setObjectName("cardTitle")
        log_layout.addWidget(log_title)
        self.vid_log = QTextEdit()
        self.vid_log.setObjectName("logBox")
        self.vid_log.setReadOnly(True)
        self.vid_log.setMaximumHeight(220)
        log_layout.addWidget(self.vid_log)
        right.addWidget(log_card)

        body.addLayout(right, 1)
        layout.addLayout(body)
        return page

    def _mini_card(self, title, value, color):
        card = self._card_widget()
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignCenter)
        t = QLabel(title)
        t.setStyleSheet("color: #64748b; font-size: 16px;")
        t.setAlignment(Qt.AlignCenter)
        v = QLabel(value)
        v.setStyleSheet(f"color: {color}; font-size: 32px; font-weight: bold;")
        v.setAlignment(Qt.AlignCenter)
        if title == "人员":
            self.mini_person = v
        elif title == "斧头":
            self.mini_axe = v
        elif title == "电锯":
            self.mini_chainsaw = v
        layout.addWidget(t)
        layout.addWidget(v)
        return card

    def _run_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if not file:
            return

        self._stop_video_internal()

        self.vid_time_label.setText(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
        self._log(self.vid_log, f"加载视频: {os.path.basename(file)}")

        # 获取视频信息并初始化保存
        cap = cv2.VideoCapture(file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        save_dir = r"D:\forest_detection_system\results_video"
        os.makedirs(save_dir, exist_ok=True)
        self._video_save_path = os.path.join(
            save_dir,
            f"video_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(self._video_save_path, fourcc, fps, (width, height))

        try:
            self.video_generator = detect_video(file, save=False, for_ui=True)
        except Exception as e:
            self._log(self.vid_log, f"加载视频失败: {e}")
            self._release_video_writer()
            return

        self.total_frames = 0
        self.detected_person = 0
        self.detected_axe = 0
        self.detected_chainsaw = 0
        self._last_risk = None
        self._reset_mini_stats()

        self.timer.start(30)
        self.btn_pause_vid.setEnabled(True)
        self.btn_stop_vid.setEnabled(True)
        self.paused = False
        self.btn_pause_vid.setText("暂停")
        self.progress_bar.setValue(0)

    def _toggle_pause(self):
        if self.paused:
            self.timer.start(30)
            self.btn_pause_vid.setText("暂停")
            self._log(self.vid_log, "继续播放")
        else:
            self.timer.stop()
            self.btn_pause_vid.setText("继续")
            self._log(self.vid_log, "已暂停")
        self.paused = not self.paused

    def _stop_video(self):
        self._stop_video_internal()
        self.video_display.setText("选择视频开始实时检测")
        self.video_display.setStyleSheet("color: #94a3b8; font-size: 18px;")
        self.vid_risk_big.setText("SAFE")
        self.vid_risk_big.setStyleSheet("""
            QLabel {
                background: #f0fdf4;
                color: #16a34a;
                border: 2px solid #bbf7d0;
                border-radius: 24px;
                padding: 36px;
                font-size: 52px;
                font-weight: bold;
            }
        """)
        self._log(self.vid_log, "检测已停止")
        self.vid_frame_info.setText("帧: 0 | 状态: 待机")
        self.progress_bar.setValue(0)

    def _stop_video_internal(self):
        """安全停止视频播放，不重置 UI（内部使用）"""
        if self.timer.isActive():
            self.timer.stop()
        self.video_generator = None
        self.btn_pause_vid.setEnabled(False)
        self.btn_stop_vid.setEnabled(False)
        self.paused = False
        self.btn_pause_vid.setText("暂停")
        self._release_video_writer()

    def _release_video_writer(self):
        """释放视频写入器并记录保存路径"""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            if self._video_save_path and os.path.exists(self._video_save_path):
                self._log(self.vid_log, f"视频已保存: {self._video_save_path}")

    def next_frame(self):
        if self.video_generator is None:
            return
        try:
            frame, results = next(self.video_generator)
            self.total_frames += 1

            # 写入保存视频
            if self._video_writer is not None:
                self._video_writer.write(frame)

            self.display_image(frame, self.video_display)

            risk = self._calculate_risk(results)
            self._update_video_risk(risk)

            # 统计
            if results and results[0].boxes:
                boxes = results[0].boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                labels = [results[0].names[i] for i in cls_ids]
                self.detected_person += labels.count("person")
                self.detected_axe += labels.count("axe")
                self.detected_chainsaw += labels.count("chainsaw")

            self.mini_person.setText(str(self.detected_person))
            self.mini_axe.setText(str(self.detected_axe))
            self.mini_chainsaw.setText(str(self.detected_chainsaw))

            self.vid_frame_info.setText(
                f"帧: {self.total_frames} | 状态: {'暂停' if self.paused else '检测中'}"
            )

            # 模拟进度
            if self.total_frames % 10 == 0:
                val = (self.progress_bar.value() + 2) % 100
                self.progress_bar.setValue(val)

        except StopIteration:
            self._stop_video_internal()
            self._release_video_writer()
            self.progress_bar.setValue(100)
            self._log(self.vid_log, "视频检测完成")
            self._log(self.vid_log, f"视频已保存: {self._video_save_path}")
            self.vid_frame_info.setText(f"帧: {self.total_frames} | 状态: 完成")
        except Exception as e:
            self._stop_video_internal()
            self._release_video_writer()
            self._log(self.vid_log, f"播放异常: {e}")
            traceback.print_exc()

    def _update_video_risk(self, risk):
        if risk == "SAFE":
            self.vid_risk_big.setText("SAFE")
            self.vid_risk_big.setStyleSheet("""
                QLabel {
                    background: #f0fdf4;
                    color: #16a34a;
                    border: 2px solid #bbf7d0;
                    border-radius: 24px;
                    padding: 36px;
                    font-size: 52px;
                    font-weight: bold;
                }
            """)
        elif risk == "MEDIUM":
            self.vid_risk_big.setText("MEDIUM")
            self.vid_risk_big.setStyleSheet("""
                QLabel {
                    background: #fff7ed;
                    color: #ea580c;
                    border: 2px solid #fed7aa;
                    border-radius: 24px;
                    padding: 36px;
                    font-size: 52px;
                    font-weight: bold;
                }
            """)
            if self._last_risk != "MEDIUM":
                self._log(self.vid_log, "警告: 检测到中等风险!")
        else:
            self.vid_risk_big.setText("HIGH RISK")
            self.vid_risk_big.setStyleSheet("""
                QLabel {
                    background: #fef2f2;
                    color: #dc2626;
                    border: 2px solid #fecaca;
                    border-radius: 24px;
                    padding: 36px;
                    font-size: 52px;
                    font-weight: bold;
                }
            """)
            if self._last_risk != "HIGH":
                self._log(self.vid_log, "警告: 检测到高风险!!!")
        self._last_risk = risk
        self._update_stats(risk)

    def _calculate_risk(self, results):
        if results is None:
            return "SAFE"
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return "SAFE"
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        labels = [results[0].names[i] for i in cls_ids]
        has_person = "person" in labels
        has_axe = "axe" in labels
        has_chain = "chainsaw" in labels
        if has_person and has_chain:
            return "HIGH"
        if has_person and has_axe:
            return "MEDIUM"
        return "SAFE"

    def _risk_style(self, person, axe, chainsaw):
        if person and chainsaw:
            return "HIGH RISK", """
                QLabel {
                    background: #fef2f2;
                    color: #dc2626;
                    border: 1px solid #fecaca;
                    border-radius: 14px;
                    padding: 18px;
                    font-size: 18px;
                    font-weight: bold;
                }
            """
        if person and axe:
            return "MEDIUM RISK", """
                QLabel {
                    background: #fff7ed;
                    color: #ea580c;
                    border: 1px solid #fed7aa;
                    border-radius: 14px;
                    padding: 18px;
                    font-size: 18px;
                    font-weight: bold;
                }
            """
        return "SAFE", """
            QLabel {
                background: #f0fdf4;
                color: #16a34a;
                border: 1px solid #bbf7d0;
                border-radius: 14px;
                padding: 18px;
                font-size: 18px;
                font-weight: bold;
            }
        """

    # ============================ 通用工具 ============================
    def display_image(self, frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qt_img)
        label.setPixmap(
            pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _log(self, text_edit, message):
        ts = datetime.now().strftime("%H:%M:%S")
        text_edit.append(f"[{ts}] {message}")
        scrollbar = text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_stats(self, risk):
        if not hasattr(self, 'val_total'):
            return
        total = int(self.val_total.text()) + 1
        self.val_total.setText(str(total))
        if risk == "SAFE":
            self.val_safe.setText(str(int(self.val_safe.text()) + 1))
        elif risk == "MEDIUM":
            self.val_warning.setText(str(int(self.val_warning.text()) + 1))
        else:
            self.val_danger.setText(str(int(self.val_danger.text()) + 1))

    def _reset_mini_stats(self):
        self.mini_person.setText("0")
        self.mini_axe.setText("0")
        self.mini_chainsaw.setText("0")

    def _animate_entry(self):
        self.nav_panel.setGeometry(-320, 0, 320, self.height())
        anim = QPropertyAnimation(self.nav_panel, b"pos")
        anim.setDuration(600)
        anim.setStartValue(QPoint(-320, 0))
        anim.setEndValue(QPoint(0, 0))
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()

    # ============================ 安全退出 ============================
    def closeEvent(self, event):
        """确保退出时释放所有资源"""
        self._stop_video_internal()
        if self.clock_timer.isActive():
            self.clock_timer.stop()
        event.accept()


# ============================================================
#  启动入口
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 12))
    win = AdvancedUI()
    win.show()
    sys.exit(app.exec_())
