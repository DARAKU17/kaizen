import sys
import requests

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QLineEdit, QPushButton,
    QListWidget, QLabel, QFrame
)
from PyQt6.QtCore import Qt


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "Phi-3.5"
SYSTEM_PROMPT = """
You are Kaizen.

You are a conversational AI, not a textbook.

STYLE RULES:
- Keep responses natural and human-like
- Prefer short paragraphs over long explanations
- Avoid long lists unless user explicitly asks
- Never sound like a lecture or Wikipedia page
- If user is vague, ask a simple clarifying question instead of over-explaining
- Do not overwhelm with information

BEHAVIOR:
- Respond like a smart assistant talking in real time
- Focus on what the user likely means, not all possible meanings
- Keep tone calm, grounded, slightly casual
"""

# -------------------------
# MAIN APP
# -------------------------
class KaizenUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Kaizen Interface 🧠")
        self.setGeometry(100, 100, 1100, 700)

        self.setStyleSheet("""
            QWidget {
                background-color: #0f1116;
                color: #e6e6e6;
                font-size: 14px;
            }

            QTextEdit {
                background-color: #151922;
                border: none;
                padding: 12px;
                border-radius: 10px;
            }

            QLineEdit {
                background-color: #151922;
                border: 1px solid #2a2f3a;
                padding: 10px;
                border-radius: 10px;
            }

            QPushButton {
                background-color: #2b6cff;
                border-radius: 10px;
                padding: 10px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #3b7cff;
            }

            QListWidget {
                background-color: #0c0e13;
                border: none;
            }
        """)

        # -------------------------
        # LAYOUTS
        # -------------------------
        root = QHBoxLayout(self)

        # SIDEBAR
        self.sidebar = QListWidget()
        self.sidebar.addItem("💬 New Chat")
        self.sidebar.setFixedWidth(200)

        # CHAT AREA
        chat_container = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Message Kaizen...")

        self.send_btn = QPushButton("Send")

        # bottom input row
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_box)
        input_row.addWidget(self.send_btn)

        chat_container.addWidget(self.chat_display)
        chat_container.addLayout(input_row)

        # add to root
        root.addWidget(self.sidebar)
        root.addLayout(chat_container)

        # -------------------------
        # SIGNALS
        # -------------------------
        self.send_btn.clicked.connect(self.send)
        self.input_box.returnPressed.connect(self.send)

    # -------------------------
    # UI HELPERS
    # -------------------------
    def add_message(self, sender, text):
        if sender == "You":
            self.chat_display.append(f"\n🧍 <b>You:</b> {text}")
        else:
            self.chat_display.append(f"\n🧠 <b>Kaizen:</b> {text}")

    # -------------------------
    # OLLAMA CALL
    # -------------------------
    def query_ollama(self, prompt):
        full_prompt = f"""
    {SYSTEM_PROMPT}
    
    User: {prompt}
    Kaizen:
    """
    
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
    
            data = res.json()
    
            if "response" not in data:
                return f"[ERROR] Bad response format: {data}"
    
            return data["response"].strip()
    
        except Exception as e:
            return f"[ERROR] {e}"

    # -------------------------
    # SEND LOGIC
    # -------------------------
    def send(self):
        text = self.input_box.text().strip()
        if not text:
            return

        self.add_message("You", text)
        self.input_box.clear()

        self.add_message("Kaizen", "thinking...")

        response = self.query_ollama(text)

        # replace last "thinking..."
        content = self.chat_display.toPlainText().rsplit("Kaizen: thinking...", 1)[0]
        self.chat_display.setPlainText(content)

        self.add_message("Kaizen", response)


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KaizenUI()
    window.show()
    sys.exit(app.exec())