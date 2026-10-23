"""Заглушки Telegram-объектов для тестов обработчиков (без сети и токена)."""
from types import SimpleNamespace


class FakeState:
    def __init__(self, state=None, data=None):
        self.state = getattr(state, "state", state)
        self.data = dict(data or {})

    async def get_state(self):
        return self.state

    async def set_state(self, state):
        self.state = getattr(state, "state", state)

    async def clear(self):
        self.state, self.data = None, {}

    async def get_data(self):
        return dict(self.data)

    async def update_data(self, **kwargs):
        self.data.update(kwargs)


class FakeMessage:
    def __init__(self, chat_id=1, **attrs):
        self.chat = SimpleNamespace(id=chat_id)
        self.answers: list[str] = []
        self.markups: list = []
        self.keyboard_removed = False
        self.photo = attrs.get("photo")
        self.document = attrs.get("document")
        self.text = attrs.get("text")

    async def answer(self, text, reply_markup=None, **kwargs):
        self.answers.append(text)
        self.markups.append(reply_markup)

    async def edit_reply_markup(self, reply_markup=None, **kwargs):
        self.keyboard_removed = reply_markup is None

    @property
    def last(self):
        return self.answers[-1]


class FakeCallback:
    def __init__(self, data, message=None):
        self.data = data
        self.message = message or FakeMessage()
        self.alerts: list[tuple] = []
        self.answered = 0

    async def answer(self, text=None, show_alert=False, **kwargs):
        self.answered += 1
        if text:
            self.alerts.append((text, show_alert))
