from __future__ import annotations

import argparse
import asyncio
import http
import logging
import socket
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import msgpack_numpy
import torch
import websockets
import websockets.asyncio.server as _server
import websockets.frames

from q_select import QSelector


LOGGER = logging.getLogger(__name__)


def _obs_images(obs: dict, views: list[str]) -> list[np.ndarray]:
    images = []
    for view in views:
        if view in {"image", "base", "agentview"}:
            key = "observation/image"
        elif view in {"wrist", "wrist_image"}:
            key = "observation/wrist_image"
        else:
            key = view
        if key in obs:
            images.append(obs[key])
    return images


class QSelectPolicy:
    def __init__(
        self,
        *,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: str | None,
        critic_path: str | None,
        num_action_samples: int,
        sample_mode: str,
        q_views: list[str],
        seed: int,
    ) -> None:
        self.train_config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(
            self.train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
        )
        self.num_action_samples = max(1, int(num_action_samples))
        self.sample_mode = sample_mode
        self.q_views = q_views
        self.rng = np.random.default_rng(seed)
        self.selector = QSelector(critic_path) if critic_path else None

        model_config = self.train_config.model
        self.action_horizon = int(getattr(model_config, "action_horizon", 10))
        self.action_dim = int(getattr(model_config, "action_dim", 32))

        if self.sample_mode in {"qselect", "best"} and self.selector is None:
            raise ValueError("--critic-path is required when --sample-mode=qselect/best")

    def _sample_noise_batch(self, sample_count: int) -> np.ndarray:
        return self.rng.normal(size=(sample_count, self.action_horizon, self.action_dim)).astype(np.float32)

    @property
    def metadata(self) -> dict:
        base = getattr(self.policy, "metadata", {}) or {}
        return {
            **base,
            "ours_qselect": {
                "num_action_samples": self.num_action_samples,
                "sample_mode": self.sample_mode,
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "batched_policy_sampling": True,
            },
        }

    def _sample_noise(self) -> np.ndarray:
        return self.rng.normal(size=(self.action_horizon, self.action_dim)).astype(np.float32)

    def _infer_candidate(self, obs: dict, use_noise: bool) -> np.ndarray:
        if use_noise:
            try:
                return np.asarray(self.policy.infer(obs, noise=self._sample_noise())["actions"], dtype=np.float32)
            except TypeError as exc:
                LOGGER.warning("Policy rejected explicit noise (%s); falling back to policy RNG.", exc)
        return np.asarray(self.policy.infer(obs)["actions"], dtype=np.float32)

    def _batch_jax_inputs(self, inputs: dict, sample_count: int) -> dict:
        return jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x)[None, ...], sample_count, axis=0), inputs)

    def _batch_torch_inputs(self, inputs: dict, sample_count: int) -> dict:
        device = self.policy._pytorch_device  # noqa: SLF001

        def convert(x):
            tensor = torch.from_numpy(np.asarray(x)).to(device)
            return tensor.unsqueeze(0).repeat((sample_count,) + (1,) * tensor.ndim)

        return jax.tree.map(convert, inputs)

    def _output_transform_batch(self, outputs: dict) -> np.ndarray:
        actions = np.asarray(outputs["actions"])
        states = np.asarray(outputs["state"])
        transformed = []
        for i in range(actions.shape[0]):
            row = self.policy._output_transform({"state": states[i], "actions": actions[i]})  # noqa: SLF001
            transformed.append(np.asarray(row["actions"], dtype=np.float32))
        return np.stack(transformed, axis=0)

    def _infer_candidates_batched(self, obs: dict, sample_count: int) -> tuple[np.ndarray, dict]:
        timing: dict[str, float | bool] = {"batched": True}

        preprocess_start = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.policy._input_transform(inputs)  # noqa: SLF001
        timing["preprocess_ms"] = (time.monotonic() - preprocess_start) * 1000.0

        sample_kwargs = dict(self.policy._sample_kwargs)  # noqa: SLF001
        use_explicit_noise = sample_count > 1

        model_start = time.monotonic()
        if self.policy._is_pytorch_model:  # noqa: SLF001
            batched_inputs = self._batch_torch_inputs(inputs, sample_count)
            rng_or_device = self.policy._pytorch_device  # noqa: SLF001
            if use_explicit_noise:
                noise = torch.from_numpy(self._sample_noise_batch(sample_count)).to(rng_or_device)
                sample_kwargs["noise"] = noise
            observation = _model.Observation.from_dict(batched_inputs)
            raw_actions = self.policy._sample_actions(rng_or_device, observation, **sample_kwargs)  # noqa: SLF001
            outputs = {
                "state": batched_inputs["state"].detach().cpu().numpy(),
                "actions": raw_actions.detach().cpu().numpy(),
            }
        else:
            batched_inputs = self._batch_jax_inputs(inputs, sample_count)
            self.policy._rng, rng = jax.random.split(self.policy._rng)  # noqa: SLF001
            if use_explicit_noise:
                sample_kwargs["noise"] = jnp.asarray(self._sample_noise_batch(sample_count))
            observation = _model.Observation.from_dict(batched_inputs)
            raw_actions = self.policy._sample_actions(rng, observation, **sample_kwargs)  # noqa: SLF001
            outputs = jax.tree.map(lambda x: np.asarray(x), {"state": batched_inputs["state"], "actions": raw_actions})
        timing["model_ms"] = (time.monotonic() - model_start) * 1000.0

        output_start = time.monotonic()
        candidate_actions = self._output_transform_batch(outputs)
        timing["output_transform_ms"] = (time.monotonic() - output_start) * 1000.0
        return candidate_actions.astype(np.float32), timing

    def _infer_candidates_fallback(self, obs: dict, sample_count: int) -> tuple[np.ndarray, dict]:
        start = time.monotonic()
        candidates = []
        for sample_index in range(sample_count):
            candidates.append(self._infer_candidate(obs, use_noise=sample_index > 0))
        return np.stack(candidates, axis=0).astype(np.float32), {
            "batched": False,
            "fallback_repeated_infer_ms": (time.monotonic() - start) * 1000.0,
        }

    def infer(self, obs: dict) -> dict:
        start = time.monotonic()
        sample_count = 1 if self.sample_mode == "first" else self.num_action_samples
        try:
            candidate_actions, policy_timing = self._infer_candidates_batched(obs, sample_count)
        except TypeError as exc:
            LOGGER.warning("Batched explicit-noise sampling failed (%s); falling back to repeated policy.infer.", exc)
            candidate_actions, policy_timing = self._infer_candidates_fallback(obs, sample_count)

        if self.sample_mode == "first" or self.num_action_samples == 1:
            selected_index = 0
            q_values = np.zeros((candidate_actions.shape[0],), dtype=np.float32)
        elif self.sample_mode == "random":
            selected_index = int(self.rng.integers(0, candidate_actions.shape[0]))
            q_values = np.zeros((candidate_actions.shape[0],), dtype=np.float32)
        else:
            assert self.selector is not None
            qselect_start = time.monotonic()
            images = _obs_images(obs, self.q_views)
            state = np.asarray(obs["observation/state"], dtype=np.float32)
            prompt = str(obs.get("prompt", ""))
            selected_index, q_values = self.selector.select(images, prompt, state, candidate_actions, mode="qselect")
            policy_timing["qselect_ms"] = (time.monotonic() - qselect_start) * 1000.0

        policy_timing["ours_total_ms"] = (time.monotonic() - start) * 1000.0
        return {
            "actions": candidate_actions[selected_index],
            "candidate_actions": candidate_actions,
            "selected_index": np.int64(selected_index),
            "q_values": np.asarray(q_values, dtype=np.float32),
            "sample_mode": self.sample_mode,
            "policy_timing": policy_timing,
        }


class QSelectWebsocketServer:
    def __init__(self, policy: QSelectPolicy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        LOGGER.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))

        prev_total_time = None
        while True:
            try:
                start = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                infer_start = time.monotonic()
                result = self.policy.infer(obs)
                result["server_timing"] = {"infer_ms": (time.monotonic() - infer_start) * 1000.0}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start
            except websockets.ConnectionClosed:
                LOGGER.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--critic-path", default="")
    parser.add_argument("--default-prompt", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-action-samples", type=int, default=8)
    parser.add_argument("--sample-mode", choices=("qselect", "best", "random", "first"), default="qselect")
    parser.add_argument("--q-views", default="image,wrist_image")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    policy = QSelectPolicy(
        config_name=args.config,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt or None,
        critic_path=args.critic_path or None,
        num_action_samples=args.num_action_samples,
        sample_mode=args.sample_mode,
        q_views=[x.strip() for x in args.q_views.split(",") if x.strip()],
        seed=args.seed,
    )
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    LOGGER.info("Creating qselect policy server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)
    QSelectWebsocketServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
