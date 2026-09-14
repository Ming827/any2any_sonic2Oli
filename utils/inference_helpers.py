"""Export trained RL policies to ONNX format for deployment.

Focuses on SONIC / universal-token model export:
  - encoder+decoder pair    (``export_universal_token_module_as_onnx``)
  - all encoders             (``export_universal_token_encoders_as_onnx``)
  - decoder only             (``export_universal_token_decoder_as_onnx``)
  - generic policy           (``export_policy_as_onnx``)
  - flat-input LoRA wrapper  (``export_oli_lora_as_onnx``)
"""

import copy
import os

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Generic policy export
# ---------------------------------------------------------------------------


def export_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
    """Export a PPO actor policy as an ONNX model.

    Args:
        inference_model: Dict containing an ``"actor"`` key with the actor module.
        path: Directory path to save the exported model.
        exported_policy_name: Filename for the exported ONNX model.
        example_obs_dict: Example observation dict for ONNX tracing.
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model["actor"]).to("cpu")
    actor.eval()

    class PPOWrapper(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs_dict):
            return self.actor.act_inference(obs_dict)

    wrapper = PPOWrapper(actor)
    example_input_list = {"obs_dict": example_obs_dict}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_list,
            path,
            verbose=True,
            input_names=["obs_dict"],
            output_names=["action"],
            opset_version=13,
        )


# ---------------------------------------------------------------------------
# Universal-token encoder + decoder pair
# ---------------------------------------------------------------------------


def export_universal_token_module_as_onnx(
    universal_token_module, encoder_name, decoder_name, path, exported_model_name, batch_size=1
):
    """Export UniversalTokenModule with a specific encoder and decoder as ONNX.

    The exported model accepts a single flattened 2-D tensor whose layout is::

        [tokenizer_observations | proprioception]

    Tokenizer observations are ordered according to the union of features
    required by the chosen encoder and decoder.

    Args:
        universal_token_module: The UniversalTokenModule instance.
        encoder_name: Name of the encoder to use.
        decoder_name: Name of the decoder to use.
        path: Directory path to save the ONNX model.
        exported_model_name: Name of the exported ONNX file.
        batch_size: Batch size for example input.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, exported_model_name)

    module = copy.deepcopy(universal_token_module).to("cpu")
    module.eval()

    # Determine which tokenizer observations are needed (union of encoder and decoder inputs)
    encoder_input_features = module.encoder_input_features[encoder_name]
    decoder_input_features = module.decoder_input_features[decoder_name]

    special_keys = {"token", "token_flattened", "proprioception", "action", "meta_action"}
    encoder_tokenizer_obs = [f for f in encoder_input_features if f not in special_keys]
    decoder_tokenizer_obs = [f for f in decoder_input_features if f not in special_keys]

    required_tokenizer_obs = list(set(encoder_tokenizer_obs + decoder_tokenizer_obs))

    # Calculate total input dimension
    total_feature_dim = 0
    for feature_name in required_tokenizer_obs:
        if feature_name in module.tokenizer_obs_dims:
            feature_dims = module.tokenizer_obs_dims[feature_name]
            total_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

    proprioception_dim = module.obs_dim_dict.get("actor_obs", 0)
    total_feature_dim += proprioception_dim

    example_input = torch.randn(batch_size, total_feature_dim, device="cpu")

    class UniversalTokenWrapper(nn.Module):
        def __init__(self, module, encoder_name, decoder_name, required_tokenizer_obs):
            super().__init__()
            self.module = module
            self.encoder_name = encoder_name
            self.decoder_name = decoder_name
            self.required_tokenizer_obs = required_tokenizer_obs

        def forward(self, obs_dict):
            combined_input = obs_dict["actor_obs"]
            combined_input = combined_input.unsqueeze(1)  # add sequence dimension

            tokenizer_feature_dim = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = self.module.tokenizer_obs_dims[feature_name]
                tokenizer_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

            tokenizer_part = combined_input[..., :tokenizer_feature_dim]
            proprioception_part = combined_input[..., tokenizer_feature_dim:]

            # Reconstruct tokenizer_obs dict from flattened tensor
            tokenizer_obs = {}
            index = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = tuple(self.module.tokenizer_obs_dims[feature_name])
                feature_size = torch.prod(torch.tensor(feature_dims)).item()
                feature_data = tokenizer_part[..., index : index + feature_size]
                tokenizer_obs[feature_name] = feature_data.reshape(
                    feature_data.shape[:2] + feature_dims
                )
                index += feature_size

            # Get encoder-specific observations
            encoder_input_features = self.module.encoder_input_features[self.encoder_name]
            encoder_tokenizer_obs = {
                k: tokenizer_obs[k] for k in encoder_input_features if k in tokenizer_obs
            }

            # Encode
            encoded_tokens, _ = self.module.encode(self.encoder_name, encoder_tokenizer_obs)
            encoded_tokens = encoded_tokens.unsqueeze(1)  # add back the sequence dimension

            # Prepare decode input
            decode_input_dict = {
                "token": encoded_tokens,
                "token_flattened": encoded_tokens.flatten(start_dim=-2),
                "proprioception": proprioception_part,
            }
            decode_input_dict.update(tokenizer_obs)

            # Decode
            decoded_output = self.module.decode(self.decoder_name, decode_input_dict)

            if "action" in decoded_output:
                return decoded_output["action"].squeeze(1)
            else:
                output = torch.cat(list(decoded_output.values()), dim=-1)
                return output.squeeze(1)

    wrapper = UniversalTokenWrapper(module, encoder_name, decoder_name, required_tokenizer_obs)

    obs_dict = {"actor_obs": example_input}
    example_input_list = {"obs_dict": obs_dict}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_list,
            full_path,
            verbose=True,
            input_names=["obs_dict"],
            output_names=["action"],
            opset_version=13,
        )

    print(f"\nExported ONNX model: {encoder_name} encoder -> {decoder_name} decoder")  # noqa: T201
    print(f"Saved to: {full_path}")  # noqa: T201
    print(f"Required tokenizer observations: {required_tokenizer_obs}")  # noqa: T201
    print(  # noqa: T201
        f"Input shape: {example_input.shape} "
        f"(tokenizer: {total_feature_dim - proprioception_dim}, proprioception: {proprioception_dim})"
    )


# ---------------------------------------------------------------------------
# Encoders-only export (all encoders, dynamic selection via encoder_index)
# ---------------------------------------------------------------------------


def export_universal_token_encoders_as_onnx(
    universal_token_module, path, exported_model_name, batch_size=1
):
    """Export only the ENCODERS of a UniversalTokenModule as ONNX.

    Uses ``encoder_index`` to dynamically select which encoder to run.

    Input layout::

        [encoder_index(1) | tokenizer_observations]

    Args:
        universal_token_module: The UniversalTokenModule instance.
        path: Directory path to save the ONNX model.
        exported_model_name: Name of the exported ONNX file.
        batch_size: Batch size for example input.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, exported_model_name)

    module = copy.deepcopy(universal_token_module).to("cpu")
    module.eval()

    encoder_names = module.encoders_to_iterate
    if not encoder_names:
        raise ValueError("No encoders found in the module")

    special_keys = {"token", "token_flattened", "proprioception", "action", "meta_action"}

    # Start with encoder_index, then follow module.tokenizer_obs_names order
    all_tokenizer_obs = ["encoder_index"]

    features_needed = set()
    for enc_name in encoder_names:
        encoder_input_features = module.encoder_input_features[enc_name]
        features_needed.update([f for f in encoder_input_features if f not in special_keys])

    for obs_name in module.tokenizer_obs_names:
        if obs_name in features_needed:
            all_tokenizer_obs.append(obs_name)

    required_tokenizer_obs = all_tokenizer_obs

    # Calculate tokenizer input dimension (only required observations)
    tokenizer_feature_dim = 0
    for feature_name in required_tokenizer_obs:
        if feature_name in module.tokenizer_obs_dims:
            feature_dims = module.tokenizer_obs_dims[feature_name]
            tokenizer_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

    example_tokenizer_obs = torch.randn(batch_size, tokenizer_feature_dim, device="cpu")
    example_encoder_index = torch.zeros((batch_size, 1), device="cpu")

    class EncodersOnlyWrapper(nn.Module):
        def __init__(self, module, encoder_names, required_tokenizer_obs):
            super().__init__()
            self.module = module
            self.encoder_names = encoder_names
            self.required_tokenizer_obs = required_tokenizer_obs

        def forward(self, obs_dict):
            inputs = obs_dict["actor_obs"]
            encoder_index = inputs[..., 0].long()
            encoder_index_onehot = torch.zeros(
                (encoder_index.shape[0], len(self.encoder_names)), device="cpu"
            )
            encoder_index_onehot[torch.arange(encoder_index.shape[0]), encoder_index] = 1.0
            tokenizer_input = inputs[..., 1:]

            tokenizer_input = tokenizer_input.unsqueeze(1)  # add sequence dimension (B, 1, F)

            # Reconstruct required tokenizer observations only
            tokenizer_obs = {}
            index = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = tuple(self.module.tokenizer_obs_dims[feature_name])
                feature_size = torch.prod(torch.tensor(feature_dims)).item()
                feature_data = tokenizer_input[..., index : index + feature_size]
                tokenizer_obs[feature_name] = feature_data.reshape(
                    feature_data.shape[:2] + feature_dims
                )
                index += feature_size

            all_encoded_tokens = []
            for encoder_name in self.encoder_names:
                encoder_input_features = self.module.encoder_input_features[encoder_name]
                encoder_tokenizer_obs = {
                    k: tokenizer_obs[k] for k in encoder_input_features if k in tokenizer_obs
                }
                encoded_tokens, _ = self.module.encode(encoder_name, encoder_tokenizer_obs)
                all_encoded_tokens.append(encoded_tokens)

            # Stack all encoded tokens: (num_encoders, B, token_features...)
            stacked_tokens = torch.stack(all_encoded_tokens, dim=0)

            # Use encoder_index to select the appropriate tokens via weighted sum
            encoder_weights = encoder_index_onehot.t()  # (num_encoders, B)
            for _ in range(len(stacked_tokens.shape) - 2):
                encoder_weights = encoder_weights.unsqueeze(-1)

            weighted_tokens = stacked_tokens * encoder_weights
            selected_tokens = weighted_tokens.sum(dim=0)  # (B, token_features...)
            selected_tokens = selected_tokens.flatten(start_dim=-2)

            return selected_tokens

    wrapper = EncodersOnlyWrapper(module, encoder_names, required_tokenizer_obs)

    obs_dict = {
        "actor_obs": torch.cat([example_encoder_index, example_tokenizer_obs], dim=-1),
    }
    example_input_dict = {"obs_dict": obs_dict}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_dict,
            full_path,
            verbose=True,
            input_names=["obs_dict"],
            output_names=["encoded_tokens"],
            opset_version=13,
        )

    print(  # noqa: T201
        f"\nExported ENCODERS ONLY ONNX model with {len(encoder_names)} encoders: {encoder_names}"
    )
    print(f"Saved to: {full_path}")  # noqa: T201
    print(f"Required tokenizer observations: {required_tokenizer_obs}")  # noqa: T201
    print(  # noqa: T201
        f"Input shapes: tokenizer_obs={example_tokenizer_obs.shape}, "
        f"encoder_index={example_encoder_index.shape}"
    )


# ---------------------------------------------------------------------------
# Decoder-only export
# ---------------------------------------------------------------------------


def export_universal_token_decoder_as_onnx(
    universal_token_module, decoder_name, path, exported_model_name, batch_size=1
):
    """Export only the DECODER of a UniversalTokenModule as ONNX.

    Input layout::

        [encoded_tokens | tokenizer_observations | proprioception]

    Args:
        universal_token_module: The UniversalTokenModule instance.
        decoder_name: Name of the decoder to use.
        path: Directory path to save the ONNX model.
        exported_model_name: Name of the exported ONNX file.
        batch_size: Batch size for example input.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, exported_model_name)

    module = copy.deepcopy(universal_token_module).to("cpu")
    module.eval()

    # Determine which tokenizer observations the decoder needs
    decoder_input_features = module.decoder_input_features[decoder_name]
    special_keys = {"token", "token_flattened", "proprioception", "action", "meta_action"}
    required_tokenizer_obs = [f for f in decoder_input_features if f not in special_keys]

    # Calculate tokenizer input dimension (only required observations)
    tokenizer_feature_dim = 0
    for feature_name in required_tokenizer_obs:
        if feature_name in module.tokenizer_obs_dims:
            feature_dims = module.tokenizer_obs_dims[feature_name]
            tokenizer_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

    proprioception_dim = module.obs_dim_dict.get("actor_obs", 0)
    token_total_dim = module.token_total_dim

    example_encoded_tokens = torch.randn(batch_size, token_total_dim, device="cpu")
    example_tokenizer = torch.randn(batch_size, tokenizer_feature_dim, device="cpu")
    example_proprioception = torch.randn(batch_size, proprioception_dim, device="cpu")

    class DecoderOnlyWrapper(nn.Module):
        def __init__(self, module, decoder_name, required_tokenizer_obs):
            super().__init__()
            self.module = module
            self.decoder_name = decoder_name
            self.required_tokenizer_obs = required_tokenizer_obs

        def forward(self, obs_dict):
            inputs = obs_dict["actor_obs"]

            proprioception_dim = self.module.obs_dim_dict["actor_obs"]
            token_total_dim = self.module.token_total_dim

            tokenizer_feature_dim = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = self.module.tokenizer_obs_dims[feature_name]
                tokenizer_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

            # Split input: [encoded_tokens | tokenizer_obs | proprioception]
            encoded_tokens = inputs[..., :token_total_dim]
            tokenizer_part = inputs[..., token_total_dim : token_total_dim + tokenizer_feature_dim]
            proprioception = inputs[..., -proprioception_dim:]

            # Add sequence dimension
            encoded_tokens = encoded_tokens.unsqueeze(1)
            tokenizer_part = tokenizer_part.unsqueeze(1)
            proprioception = proprioception.unsqueeze(1)

            # Reconstruct tokenizer observations from flat tensor
            tokenizer_obs = {}
            index = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = tuple(self.module.tokenizer_obs_dims[feature_name])
                feature_size = torch.prod(torch.tensor(feature_dims)).item()
                feature_data = tokenizer_part[..., index : index + feature_size]
                tokenizer_obs[feature_name] = feature_data.reshape(
                    feature_data.shape[:2] + feature_dims
                )
                index += feature_size

            # Prepare decode input
            decode_input_dict = {
                "token_flattened": encoded_tokens,
                "proprioception": proprioception,
            }
            decode_input_dict.update(tokenizer_obs)

            # Decode
            decoded_output = self.module.decode(self.decoder_name, decode_input_dict)

            if "action" in decoded_output:
                return decoded_output["action"].squeeze(1)
            else:
                output = torch.cat(list(decoded_output.values()), dim=-1)
                return output.squeeze(1)

    wrapper = DecoderOnlyWrapper(module, decoder_name, required_tokenizer_obs)

    obs_dict = {
        "actor_obs": torch.cat(
            [example_encoded_tokens, example_tokenizer, example_proprioception], dim=-1
        ),
    }
    example_input_dict = {"obs_dict": obs_dict}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_dict,
            full_path,
            verbose=True,
            input_names=["obs_dict"],
            output_names=["action"],
            opset_version=13,
        )

    print(f"\nExported DECODER ONLY ONNX model with decoder: {decoder_name}")  # noqa: T201
    print(f"Saved to: {full_path}")  # noqa: T201
    print(f"Required tokenizer observations: {required_tokenizer_obs}")  # noqa: T201
    print(  # noqa: T201
        f"Input shapes: encoded_tokens={example_encoded_tokens.shape}, "
        f"tokenizer={example_tokenizer.shape}, proprioception={example_proprioception.shape}"
    )


# ---------------------------------------------------------------------------
# Oli LoRA flat-input export (full wrapper as a single graph)
# ---------------------------------------------------------------------------


def export_oli_lora_as_onnx(actor, path, exported_model_name, batch_size=1):
    """Export an Actor wrapping ``OliLoRAWrapper`` as a single ONNX graph.

    The graph takes two flat tensors as positional inputs and returns the
    Oli 31-DOF action mean. All wrapper-internal logic (running_mean_std
    normalization, prop slice 31→29, hip-pitch axis decomposition, frozen
    G1 encoder + LoRA'd g1_dyn decoder, DofBridge 29→31) is baked into the
    graph — deploy code only needs to assemble the two flat tensors and
    apply ACTION_SCALE / PD downstream.

    Why a custom export function: the generic ``export_policy_as_onnx``
    routes through ``Actor.act_inference``, which mutates a stateful
    ``obs_dict_buffer``. ONNX trace cannot capture that side effect cleanly
    and the resulting graph fails to load. This wrapper sidesteps the
    buffer entirely (we add a synthetic seq_len=1 dim, run forward once,
    squeeze it back).

    Args:
        actor: The Actor module (``trl.modules.actor_critic_modules.Actor``)
            whose ``actor_module`` is an ``OliLoRAWrapper``.
        path: Output directory.
        exported_model_name: ONNX filename.
        batch_size: Example batch size for tracing.

    Inputs:
        actor_obs: ``[B, 990]`` Oli proprioception (policy obs group).
        tokenizer: ``[B, 1801]`` flat tokenizer obs in
            ``module.tokenizer_obs_names`` order.

    Output:
        action: ``[B, 31]`` Oli action mean (raw, before ACTION_SCALE).
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, exported_model_name)

    actor = copy.deepcopy(actor).to("cpu")
    actor.eval()

    actor_module = actor.actor_module
    if not hasattr(actor_module, "pretrained"):
        raise ValueError(
            "export_oli_lora_as_onnx expects an Actor wrapping OliLoRAWrapper "
            "(actor.actor_module.pretrained must exist)"
        )

    actor_obs_dim = int(actor_module.pretrained.obs_dim_dict["actor_obs"])
    tokenizer_dim = sum(
        int(torch.tensor(actor_module.pretrained.tokenizer_obs_dims[name]).prod().item())
        for name in actor_module.pretrained.tokenizer_obs_names
    )

    class FlatLoRAWrapper(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, actor_obs, tokenizer):
            if self.actor.running_mean_std is not None:
                actor_obs = self.actor.running_mean_std(actor_obs)
            input_data = {
                "actor_obs": actor_obs.unsqueeze(1),
                "tokenizer": tokenizer.unsqueeze(1),
            }
            action_seq = self.actor.actor_module(input_data)
            return action_seq.squeeze(1)

    wrapper = FlatLoRAWrapper(actor)
    example_actor_obs = torch.zeros(batch_size, actor_obs_dim, dtype=torch.float32)
    example_tokenizer = torch.zeros(batch_size, tokenizer_dim, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example_actor_obs, example_tokenizer),
            full_path,
            input_names=["actor_obs", "tokenizer"],
            output_names=["action"],
            opset_version=13,
        )

    print(f"\nExported Oli LoRA ONNX model: {full_path}")  # noqa: T201
    print(  # noqa: T201
        f"Input shapes: actor_obs={list(example_actor_obs.shape)}, "
        f"tokenizer={list(example_tokenizer.shape)}"
    )
    print(  # noqa: T201
        f"Tokenizer obs order ({len(actor_module.pretrained.tokenizer_obs_names)} fields): "
        f"{actor_module.pretrained.tokenizer_obs_names}"
    )
