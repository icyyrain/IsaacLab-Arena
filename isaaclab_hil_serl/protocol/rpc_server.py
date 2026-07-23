# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RPC server that owns a Gymnasium environment in the Windows process."""

from __future__ import annotations

import gymnasium as gym
import socketserver
import threading
from typing import Any

from .codec import receive_message, send_message, space_to_spec

PROTOCOL_VERSION = 1


class _EnvironmentTcpServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], env: gym.Env):
        self.env = env
        self.env_closed = False
        super().__init__(server_address, _RequestHandler)

    def dispatch(self, request: dict[str, Any]) -> Any:
        method = request.get("method")
        params = request.get("params", {})
        assert isinstance(params, dict), "RPC params must be a dictionary"

        if method == "ping":
            return {"protocol_version": PROTOCOL_VERSION}
        if method == "describe":
            return {
                "protocol_version": PROTOCOL_VERSION,
                "observation_space": space_to_spec(self.env.observation_space),
                "action_space": space_to_spec(self.env.action_space),
                "metadata": dict(getattr(self.env, "metadata", {})),
            }
        if method == "reset":
            return self.env.reset(seed=params.get("seed"), options=params.get("options"))
        if method == "step":
            return self.env.step(params["action"])
        if method == "close":
            self.close_env()
            return None
        raise ValueError(f"Unknown RPC method: {method!r}")

    def close_env(self) -> None:
        if not self.env_closed:
            self.env.close()
            self.env_closed = True


class _RequestHandler(socketserver.BaseRequestHandler):
    server: _EnvironmentTcpServer

    def handle(self) -> None:
        while True:
            try:
                request = receive_message(self.request)
            except (ConnectionError, EOFError):
                return

            try:
                assert isinstance(request, dict), "RPC request must be a dictionary"
                result = self.server.dispatch(request)
                response = {"ok": True, "result": result}
            except Exception as error:
                response = {
                    "ok": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            try:
                send_message(self.request, response)
            except (ConnectionError, EOFError):
                return


class EnvironmentRpcServer:
    """Expose one Gymnasium environment over a trusted-machine TCP connection."""

    def __init__(self, env: gym.Env, host: str = "127.0.0.1", port: int = 0):
        self._server = _EnvironmentTcpServer((host, port), env)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address
        return str(host), int(port)

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        assert self._thread is None, "Environment RPC server is already running"
        self._thread = threading.Thread(target=self.serve_forever, name="hil-serl-env-rpc", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.close_env()
        self._server.server_close()

    def __enter__(self) -> EnvironmentRpcServer:
        self.start_in_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
