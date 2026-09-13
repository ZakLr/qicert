"""Fork-equivalent MiniVLA batch builder without dlimp/TensorFlow.

The fork's ``prismatic.vla.datasets`` module imports its RLDS pipeline at
package top-level, which hard-requires ``dlimp`` -> ``tensorflow``; neither
installable on native py3.14 (no cp314 wheels). This module replicates the
EXACT semantics of ``RLDSBatchTransform`` + ``PaddedCollatorForActionPrediction``
(verified line-by-line against weights/code @ transformers-5.x patch) using
only safe prismatic pieces (tokenizer, ActionTokenizer, image transform,
prompt builder — all attributes of the loaded VLA or proven-importable).

A logged, documented substitution (see run metadata). Feeds from the
NPZ episode bridge instead of the upstream RLDS stream.
"""
from __future__ import annotations

import numpy as np

IGNORE_INDEX = -100
NUM_END_TOKENS = 2  # Qwen2TokenizerFast: <|im_end|><|endoftext|>


class LocalVLABatcher:
    """Builds model-ready batches identical to the fork's train/eval path."""

    def __init__(self, vla):
        import torch
        from prismatic.vla.action_tokenizer import ActionTokenizer

        self._torch = torch
        self.tokenizer = vla.llm_backbone.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        self.image_transform = vla.vision_backbone.get_image_transform()
        self.prompt_builder_fn = vla.llm_backbone.prompt_builder_fn
        self.model_max_length = int(self.tokenizer.model_max_length)
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    def example(self, frame_u8: np.ndarray, action7: np.ndarray,
                language: str) -> dict:
        """One RLDS-style sample -> transformed example (fork semantics)."""
        import torch
        from PIL import Image

        img = Image.fromarray(frame_u8)
        # horizon-0 contract: tokenizer sees the single current action
        tokenized_action = self.action_tokenizer(np.asarray(action7, dtype=np.float64))
        raw_action_tokens = self.tokenizer(tokenized_action)["input_ids"]

        pb = self.prompt_builder_fn("openvla")
        pb.add_turn("human",
                    f"What action should the robot take to {language.lower()}?")
        pb.add_turn("gpt", tokenized_action)

        input_ids_list = list(self.tokenizer(pb.get_prompt(),
                                             add_special_tokens=True).input_ids)
        # Fork masks AFTER tensorization (torch accepts scalar slice-assign).
        input_ids = torch.tensor(input_ids_list)
        labels = torch.tensor(input_ids_list)
        num_answer_tokens = len(raw_action_tokens)
        labels[:-(num_answer_tokens + NUM_END_TOKENS)] = IGNORE_INDEX

        return {
            "pixel_values": self.image_transform(img),
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": "libero_spatial_no_noops",
        }

    def collate(self, instances: list[dict]) -> dict:
        """PaddedCollatorForActionPrediction semantics."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        input_ids = pad_sequence([e["input_ids"] for e in instances],
                                 batch_first=True,
                                 padding_value=self.pad_token_id)
        labels = pad_sequence([e["labels"] for e in instances],
                              batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)
        pvs = [e["pixel_values"] for e in instances]
        if isinstance(pvs[0], dict):
            pixel_values = {kk: torch.stack([pv[kk] for pv in pvs])
                            for kk in pvs[0]}
        else:
            pixel_values = torch.stack(pvs)
        return {"pixel_values": pixel_values, "input_ids": input_ids,
                "attention_mask": attention_mask, "labels": labels}


class LocalEpisodeStream:
    """Infinite shuffled stream of collated batches over NPZ episodes.

    Drop-in replacement for iter(DataLoader(RLDSDataset...)): each next()
    returns one CPU batch dict. Episodes are visited in a reshuffled order
    every epoch and frames walked sequentially inside an episode (the hot
    episode stays decoded/resized in memory); horizon-0 single actions
    match N1's RLDS window config.
    """

    def __init__(self, dataset, batcher: LocalVLABatcher, batch_size: int,
                 seed: int = 0, episode_ids: list[int] | None = None):
        self.ds = dataset
        self.batcher = batcher
        self.bs = batch_size
        self.rng = np.random.default_rng(seed)
        # episode_ids: optional allow-list of episode indices (train-only
        # filtering for N9 repair training — keeps eval episodes out of the
        # optimizer). None = all episodes (legacy behavior).
        self._episode_ids = (list(episode_ids) if episode_ids is not None
                             else list(range(len(dataset.files))))
        self._order = self.rng.permutation(len(self._episode_ids))
        self._ep_pos = 0          # pointer into _order
        self._frames = self._acts = self._lang = None
        self._t = 0

    def _advance_episode(self) -> None:
        if self._ep_pos >= len(self._order):
            self._order = self.rng.permutation(len(self._episode_ids))
            self._ep_pos = 0
        ei = self._episode_ids[int(self._order[self._ep_pos])]
        self._ep_pos += 1
        self._frames, self._acts, self._lang = self.ds[ei]
        self._t = 0

    def __iter__(self):
        return self

    def __next__(self):
        examples = []
        while len(examples) < self.bs:
            if self._frames is None or self._t >= len(self._acts):
                self._advance_episode()
            examples.append(self.batcher.example(
                self._frames[self._t], self._acts[self._t], self._lang))
            self._t += 1
        return self.batcher.collate(examples)


class LocalEvalSplitStream:
    """Finite stream over a FROZEN episode list (results/eval_split.json).

    Yields collated batches walking the given episodes in order, then stops.
    This is the honest held-out protocol: the eval episodes are fixed by a
    hashed split file BEFORE any run, so no eval/train contamination and no
    sampler luck. Each next() returns one collated batch dict; iteration
    ends after the last eval episode is consumed (StopIteration).
    """

    def __init__(self, dataset, batcher: LocalVLABatcher, episodes: list[str],
                 batch_size: int = 4):
        self.ds = dataset
        self.batcher = batcher
        self.bs = batch_size
        name_to_idx = {p.name: i for i, p in enumerate(dataset.files)}
        missing = [e for e in episodes if e not in name_to_idx]
        if missing:
            raise FileNotFoundError(f"eval split references {len(missing)} "
                                    f"episodes missing from {dataset.root} "
                                    f"(first: {missing[0]})")
        self._episodes = [name_to_idx[e] for e in episodes]
        self._ep_pos = 0
        self._frames = self._acts = self._lang = None
        self._t = 0

    def __iter__(self):
        return self

    def __next__(self):
        examples = []
        while len(examples) < self.bs:
            if self._frames is None or self._t >= len(self._acts):
                if self._ep_pos >= len(self._episodes):
                    if examples:  # flush a final short batch
                        return self.batcher.collate(examples)
                    raise StopIteration
                self._frames, self._acts, self._lang = \
                    self.ds[self._episodes[self._ep_pos]]
                self._ep_pos += 1
                self._t = 0
            examples.append(self.batcher.example(
                self._frames[self._t], self._acts[self._t], self._lang))
            self._t += 1
        return self.batcher.collate(examples)
