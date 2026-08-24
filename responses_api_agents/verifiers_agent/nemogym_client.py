# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A verifiers v1 client that keeps the token ids NeMo Gym's model server returns.

v1 has two clients: `EvalClient` relays chat completions and carries no token ids, and
`TrainClient` talks to a raw vLLM `/inference/v1/generate` and tokenizes client-side.
Gym's model server is OpenAI-shaped but already attaches `prompt_token_ids`,
`generation_token_ids` and `generation_log_probs` — the same fields the v0
`NeMoRLChatCompletionsClient` reads. This lifts them onto `Response.tokens`, which is
what `_commit_turn` turns into per-node `token_ids` / `mask` / `logprobs`.

`TurnTokens` documents itself as "renderer tokenization *or provider-returned token IDs*",
so this is the intended seam; no renderer or tokenizer is needed.

We deliberately leave `message_spans` unset. `_commit_turn` then attributes the whole
prompt to the sampled node as its leading `gen_prompt`, masked False, with the completion
masked True. The flat training sample is exact; only per-message attribution is lost.
"""

from collections.abc import Mapping
from typing import Any

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.types import Response, TurnTokens


def _extract(raw: dict) -> tuple[list[int], list[int], list[float]] | None:
    """Pull Gym's token-id extension fields off a chat-completion body."""
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    choice = choices[0] if choices else {}

    # Gym has carried these on the message; tolerate the response/choice level too.
    prompt_ids = raw.get("prompt_token_ids") or message.get("prompt_token_ids")
    completion_ids = message.get("generation_token_ids") or choice.get("token_ids")
    logprobs = message.get("generation_log_probs")

    if not prompt_ids or not completion_ids:
        return None
    # vllm_model derives generation ids from logprob labels ("token_id:151667".removeprefix)
    # so they arrive as *strings*, while prompt ids come from the tokenize endpoint as ints.
    # Coerce explicitly rather than leaning on pydantic's implicit str->int.
    return (
        [int(t) for t in prompt_ids],
        [int(t) for t in completion_ids],
        [float(lp) for lp in (logprobs or [])],
    )


class NeMoGymClient(EvalClient):
    """`EvalClient` plus provider-returned token ids."""

    async def get_response(
        self,
        dialect: Any,
        body: dict,
        model: str,
        sampling_args: Any,
        session_id: str | None = None,
        turn: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        response = await super().get_response(
            dialect, body, model, sampling_args, session_id, turn, headers
        )
        found = _extract(response.raw or {})
        if found is None:
            # No token ids (e.g. an external provider): behave exactly like EvalClient.
            return response
        prompt_ids, completion_ids, logprobs = found
        response.tokens = TurnTokens(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=logprobs,
        )
        return response


def install() -> None:
    """Route every non-train client through `NeMoGymClient`.

    v1 resolves clients with a hard if/else in `verifiers.v1.clients.client.resolve_client`
    — there is no registry — so a third client type has to be patched in until upstream
    grows one. Consumers do `from verifiers.v1.clients import resolve_client`, binding their
    own reference, so every binding site has to be rebound, not just the defining module.
    """
    import verifiers.v1 as vf
    import verifiers.v1.clients as clients_pkg
    import verifiers.v1.clients.client as client_mod
    import verifiers.v1.interception.server as server_mod
    from verifiers.v1.configs.client import TrainClientConfig

    if getattr(client_mod.resolve_client, "_nemo_gym_client", False):
        return
    original = client_mod.resolve_client

    def resolve_client(config):
        if isinstance(config, TrainClientConfig):
            return original(config)
        return NeMoGymClient(config)

    resolve_client._nemo_gym_client = True
    for module in (client_mod, clients_pkg, server_mod, vf):
        if hasattr(module, "resolve_client"):
            module.resolve_client = resolve_client
