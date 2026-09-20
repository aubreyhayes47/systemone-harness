"""End-to-end contract check through TypeSafe's official Python client.

Skipped when the official SDK is not installed. The test starts the ASGI app on an
actual ephemeral localhost port because TypeSafeClient owns its HTTP client.
"""

import socket
import threading
import time

import pytest

pytest.importorskip("typesafe_sdk")
uvicorn = pytest.importorskip("uvicorn")

from typesafe_sdk import Choice, TypeSafeClient

from systemone_local.backends.base import ChoiceResult
from systemone_local.server import create_app


class FakeBackend:
    name = "fake-system-one"
    description = "Fake test backend"

    def choice(self, state, instructions, criteria):
        keys = list(criteria)
        return ChoiceResult(keys[0], {keys[0]: 0.75, keys[1]: 0.25}, 7)


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_official_typesafe_client_can_use_local_server():
    port = _port()
    config = uvicorn.Config(create_app(FakeBackend()), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        time.sleep(0.01)
    assert server.started

    try:
        with TypeSafeClient(
            api_key="local",
            base_url=f"http://127.0.0.1:{port}",
            model="fake-system-one",
        ) as client:
            assert client.models.list().models[0].name == "fake-system-one"
            result = client.system_one(
                state={"health": 2},
                questions={
                    "action": Choice(
                        instructions="What next?",
                        criteria={"retreat": "Retreat", "attack": "Attack"},
                    )
                },
            )
            assert result.choices["action"].choice == "retreat"
            assert result.choices["action"].probabilities["retreat"] == pytest.approx(0.75)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
