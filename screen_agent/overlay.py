"""PyQt6 transparent overlay — notification cards, action buttons, system tray."""

from __future__ import annotations

import logging
import sys
from enum import Enum, auto
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


class UserAction(Enum):
    ACCEPT = auto()
    DISMISS = auto()
    ESCALATE = auto()
    FOLLOW_UP = auto()


class NotificationCard(QWidget):
    """A rounded card that shows the agent's proposal."""

    action_clicked = pyqtSignal(UserAction, str)  # action, optional text

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("notificationCard")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self._icon_label = QLabel("🔍")
        self._icon_label.setFont(QFont("Arial", 16))
        self._title = QLabel("Screen Agent")
        self._title.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        self._title.setStyleSheet("color: #e0e0e0;")
        header.addWidget(self._icon_label)
        header.addWidget(self._title, 1)

        self._dismiss_btn = QPushButton("✕")
        self._dismiss_btn.setFixedSize(24, 24)
        self._dismiss_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #888; border: none; "
            "font-size: 14px; } QPushButton:hover { color: #fff; }"
        )
        self._dismiss_btn.clicked.connect(
            lambda: self.action_clicked.emit(UserAction.DISMISS, ""),
        )
        header.addWidget(self._dismiss_btn)
        layout.addLayout(header)

        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setFont(QFont("Helvetica Neue", 12))
        self._body.setStyleSheet("color: #ccc; line-height: 1.4;")
        layout.addWidget(self._body)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._accept_btn = self._make_button("Yes, do it", "#4CAF50")
        self._accept_btn.clicked.connect(
            lambda: self.action_clicked.emit(UserAction.ACCEPT, ""),
        )
        btn_row.addWidget(self._accept_btn)

        self._escalate_btn = self._make_button("Tell me more", "#2196F3")
        self._escalate_btn.clicked.connect(
            lambda: self.action_clicked.emit(UserAction.ESCALATE, ""),
        )
        btn_row.addWidget(self._escalate_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Conversation area (hidden by default)
        self._convo_area = QTextEdit()
        self._convo_area.setReadOnly(True)
        self._convo_area.setVisible(False)
        self._convo_area.setMinimumHeight(100)
        self._convo_area.setMaximumHeight(250)
        self._convo_area.setStyleSheet(
            "QTextEdit { background: rgba(0,0,0,0.3); color: #ddd; "
            "border: 1px solid #444; border-radius: 6px; padding: 8px; "
            "font-family: 'Helvetica Neue'; font-size: 12px; }"
        )
        layout.addWidget(self._convo_area)

        # Follow-up input (hidden by default)
        follow_row = QHBoxLayout()
        self._follow_input = QLineEdit()
        self._follow_input.setPlaceholderText("Ask a follow-up…")
        self._follow_input.setVisible(False)
        self._follow_input.setStyleSheet(
            "QLineEdit { background: rgba(255,255,255,0.08); color: #ddd; "
            "border: 1px solid #555; border-radius: 6px; padding: 6px 10px; "
            "font-size: 12px; }"
        )
        self._follow_input.returnPressed.connect(self._send_follow_up)
        follow_row.addWidget(self._follow_input)

        self._send_btn = self._make_button("Send", "#2196F3")
        self._send_btn.setVisible(False)
        self._send_btn.setFixedWidth(60)
        self._send_btn.clicked.connect(self._send_follow_up)
        follow_row.addWidget(self._send_btn)
        layout.addLayout(follow_row)

    def set_proposal(self, text: str) -> None:
        self._body.setText(text)
        self._convo_area.setVisible(False)
        self._follow_input.setVisible(False)
        self._send_btn.setVisible(False)
        self._convo_area.clear()

    def show_conversation(self, text: str) -> None:
        self._convo_area.setVisible(True)
        self._follow_input.setVisible(True)
        self._send_btn.setVisible(True)
        self._convo_area.setHtml(
            f'<div style="color: #aef;">{text}</div>'
        )

    def set_assistant_label(self, label: str) -> None:
        """Override the display name for assistant messages (e.g. 'Gemini', 'OpenAI')."""
        self._assistant_label = label

    def append_conversation(self, role: str, text: str) -> None:
        color = "#aef" if role == "assistant" else "#fea"
        label = getattr(self, "_assistant_label", "AI")
        prefix = label if role == "assistant" else "You"
        self._convo_area.append(
            f'<div style="color:{color};margin-top:6px;">'
            f"<b>{prefix}:</b> {text}</div>"
        )

    def append_stream_chunk(self, chunk: str) -> None:
        cursor = self._convo_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(chunk)
        self._convo_area.setTextCursor(cursor)
        self._convo_area.ensureCursorVisible()

    def _send_follow_up(self) -> None:
        text = self._follow_input.text().strip()
        if text:
            self._follow_input.clear()
            self.append_conversation("user", text)
            self.action_clicked.emit(UserAction.FOLLOW_UP, text)

    @staticmethod
    def _make_button(label: str, color: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFont(QFont("Helvetica Neue", 11))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background: {color}; color: white; border: none; "
            f"border-radius: 6px; padding: 6px 14px; }}"
            f"QPushButton:hover {{ background: {color}cc; }}"
        )
        return btn


class OverlayWindow(QMainWindow):
    """Frameless, translucent, always-on-top overlay with a notification card."""

    pause_toggled = pyqtSignal(bool)  # True = paused

    def __init__(self, position: str = "bottom-right", opacity: float = 0.92) -> None:
        super().__init__()
        self._position = position
        self._paused = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(380)

        central = QWidget()
        central.setStyleSheet(
            "#notificationCard { background: rgba(30, 30, 36, 0.94); "
            "border-radius: 12px; }"
        )
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self.card = NotificationCard()
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.card.setGraphicsEffect(shadow)
        root.addWidget(self.card)

        self._setup_tray()
        self.hide()

    @staticmethod
    def _make_tray_icon() -> QIcon:
        """Generate a simple eye icon for the system tray."""
        px = QPixmap(32, 32)
        px.fill(QColor(0, 0, 0, 0))
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(33, 150, 243))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(4, 8, 24, 16)
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(11, 11, 10, 10)
        p.setBrush(QColor(30, 30, 36))
        p.drawEllipse(13, 13, 6, 6)
        p.end()
        return QIcon(px)

    def _setup_tray(self) -> None:
        self._tray = QSystemTrayIcon(self._make_tray_icon(), self)
        self._tray.setToolTip("Screen Agent")

        menu = QMenu()
        self._pause_action = QAction("Pause", self)
        self._pause_action.triggered.connect(self._toggle_pause)
        menu.addAction(self._pause_action)

        show_action = QAction("Show / Hide", self)
        show_action.triggered.connect(self._toggle_visible)
        menu.addAction(show_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.show()

    def show_proposal(self, text: str) -> None:
        self.card.set_proposal(text)
        self._position_on_screen()
        self.show()
        self._raise_without_focus()

    def _raise_without_focus(self) -> None:
        """Bring overlay to front without stealing focus from the active app."""
        if sys.platform == "darwin":
            try:
                from AppKit import NSApp, NSFloatingWindowLevel  # type: ignore[import-not-found]
                for nswindow in NSApp.windows():
                    nswindow.setLevel_(NSFloatingWindowLevel)
                    nswindow.orderFrontRegardless()
                return
            except ImportError:
                pass
        self.raise_()

    def _position_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        margin = 20
        self.adjustSize()
        w, h = self.width(), self.height()

        positions = {
            "top-left": (geo.x() + margin, geo.y() + margin),
            "top-right": (geo.x() + geo.width() - w - margin, geo.y() + margin),
            "bottom-left": (geo.x() + margin, geo.y() + geo.height() - h - margin),
            "bottom-right": (
                geo.x() + geo.width() - w - margin,
                geo.y() + geo.height() - h - margin,
            ),
        }
        x, y = positions.get(self._position, positions["bottom-right"])
        self.move(int(x), int(y))

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_action.setText("Resume" if self._paused else "Pause")
        self.pause_toggled.emit(self._paused)

    def _toggle_visible(self) -> None:
        self.setVisible(not self.isVisible())

    @property
    def is_paused(self) -> bool:
        return self._paused
