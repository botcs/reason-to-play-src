"""Read supplied EMPA theory records and compute their HRR representations."""

import gzip
import json
from pathlib import Path
import re

import numpy as np

from data.values import decode_value

HRR_DIM = 348
HRR_SEED = 42


def iter_regressors(root_or_file):
    """Read EMPA theories from an explicit JSON file or dataset root."""
    path = Path(root_or_file)
    if path.is_dir():
        path = path / "analysis/neural/inputs/theory-regressors.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    if (record.get("schema"), record.get("schema_version")) != (
        "reason-to-play/empa-regressors",
        1,
    ):
        raise ValueError(f"Unsupported EMPA theory schema in {path}")
    yield from decode_value(record)["regressors"]


class HRREncoder:
    """Holographic Reduced Representation encoder for EMPA theories."""

    def __init__(self, dim: int = 348, seed: int = None):
        self.dim = dim
        self.sigma = 1.0 / np.sqrt(dim)
        self._rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self._vocab = {}
        for role in [
            "type",
            "color",
            "name",
            "effect",
            "agent",
            "patient",
            "generic",
            "rule_type",
            "sprite1",
            "sprite2",
            "outcome",
            "count",
        ]:
            self._get_or_create_vector(f"ROLE_{role}")

    def _get_or_create_vector(self, token: str) -> np.ndarray:
        if token not in self._vocab:
            self._vocab[token] = self._rng.normal(0, self.sigma, self.dim)
        return self._vocab[token]

    def _circular_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _bind(self, role: str, filler: str) -> np.ndarray:
        return self._circular_convolution(
            self._get_or_create_vector(f"ROLE_{role}"),
            self._get_or_create_vector(filler),
        )

    def _bind_with_vector(self, role: str, filler_vec: np.ndarray) -> np.ndarray:
        return self._circular_convolution(
            self._get_or_create_vector(f"ROLE_{role}"), filler_vec
        )

    def parse_theory_string(self, theory_str: str) -> dict:
        result = {"interactions": [], "terminations": [], "classes": {}}
        if not theory_str or not isinstance(theory_str, str):
            return result
        lines = theory_str.strip().split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == "InteractionSet:":
                current_section = "interactions"
            elif line == "TerminationSet:":
                current_section = "terminations"
            elif line == "Class assignments:":
                current_section = "classes"
            elif current_section == "interactions":
                match = re.match(
                    r"^(\w+)\s+(\w+)\s+(\w+)\s+(\{.*?\})\s+generic:\s*(True|False)",
                    line,
                )
                if match:
                    effect, agent, patient, params_str, generic = match.groups()
                    result["interactions"].append(
                        {
                            "effect": effect,
                            "agent": agent,
                            "patient": patient,
                            "generic": generic == "True",
                            "params": params_str,
                        }
                    )
            elif current_section == "terminations":
                parts = line.split()
                if len(parts) >= 3:
                    rule_type = parts[0]
                    if rule_type == "NoveltyRule" and len(parts) >= 4:
                        result["terminations"].append(
                            {
                                "rule_type": rule_type,
                                "sprite1": parts[1],
                                "sprite2": parts[2],
                                "explored": parts[3] if len(parts) > 3 else "True",
                                "outcome": parts[4] if len(parts) > 4 else "None",
                            }
                        )
                    elif rule_type == "SpriteCounterRule" and len(parts) >= 4:
                        result["terminations"].append(
                            {
                                "rule_type": rule_type,
                                "sprite": parts[1],
                                "count": parts[2],
                                "outcome": parts[3],
                            }
                        )
                    elif rule_type == "MultiSpriteCounterRule" and len(parts) >= 5:
                        result["terminations"].append(
                            {
                                "rule_type": rule_type,
                                "sprite1": parts[1],
                                "sprite2": parts[2],
                                "count": parts[3],
                                "outcome": parts[4],
                            }
                        )
                    else:
                        result["terminations"].append(
                            {"rule_type": rule_type, "raw": line}
                        )
            elif current_section == "classes":
                match = re.match(r"^(\w+):\s*\['(\w+)'\]:\s*<class '([^']+)'>", line)
                if match:
                    name, color, vgdl_class = match.groups()
                    vgdl_type = (
                        vgdl_class.split(".")[-1] if "." in vgdl_class else vgdl_class
                    )
                    result["classes"][name] = {"color": color, "vgdl_type": vgdl_type}
        return result

    def encode_theory(self, theory_str: str) -> np.ndarray:
        if not theory_str or not isinstance(theory_str, str):
            return np.zeros(self.dim)
        parsed = self.parse_theory_string(theory_str)
        sprite_vectors = {}
        for class_name, class_info in parsed["classes"].items():
            emb = np.zeros(self.dim)
            emb += self._bind("name", class_name)
            if "color" in class_info:
                emb += self._bind("color", class_info["color"])
            if "vgdl_type" in class_info:
                emb += self._bind("type", class_info["vgdl_type"])
            sprite_vectors[class_name] = self._normalize(emb)
        sprite_set_vec = np.zeros(self.dim)
        for sv in sprite_vectors.values():
            sprite_set_vec += sv
        sprite_set_vec = (
            self._normalize(sprite_set_vec) if sprite_vectors else sprite_set_vec
        )
        interaction_set_vec = np.zeros(self.dim)
        for inter in parsed["interactions"]:
            emb = np.zeros(self.dim)
            emb += self._bind("effect", inter["effect"])
            agent = inter["agent"]
            emb += (
                self._bind_with_vector("agent", sprite_vectors[agent])
                if agent in sprite_vectors
                else self._bind("agent", agent)
            )
            patient = inter["patient"]
            emb += (
                self._bind_with_vector("patient", sprite_vectors[patient])
                if patient in sprite_vectors
                else self._bind("patient", patient)
            )
            emb += self._bind(
                "generic",
                "generic_true" if inter.get("generic", True) else "generic_false",
            )
            interaction_set_vec += self._normalize(emb)
        interaction_set_vec = (
            self._normalize(interaction_set_vec)
            if parsed["interactions"]
            else interaction_set_vec
        )
        termination_set_vec = np.zeros(self.dim)
        for term in parsed["terminations"]:
            emb = np.zeros(self.dim)
            emb += self._bind("rule_type", term["rule_type"])
            for key in ["sprite", "sprite1"]:
                if key in term:
                    s = term[key]
                    emb += (
                        self._bind_with_vector("sprite1", sprite_vectors[s])
                        if s in sprite_vectors
                        else self._bind("sprite1", s)
                    )
            if "sprite2" in term:
                s = term["sprite2"]
                emb += (
                    self._bind_with_vector("sprite2", sprite_vectors[s])
                    if s in sprite_vectors
                    else self._bind("sprite2", s)
                )
            if "outcome" in term:
                emb += self._bind("outcome", str(term["outcome"]))
            if "count" in term:
                emb += self._bind("count", f"count_{term['count']}")
            termination_set_vec += self._normalize(emb)
        termination_set_vec = (
            self._normalize(termination_set_vec)
            if parsed["terminations"]
            else termination_set_vec
        )
        return self._normalize(
            sprite_set_vec + interaction_set_vec + termination_set_vec
        )

    def encode_theory_decomposed(self, theory_str: str) -> tuple:
        """Encode theory and return the three sub-vectors separately.

        Returns:
            (sprite_set_vec, interaction_set_vec, termination_set_vec)
            Each is a normalized vector of shape (self.dim,).
        """
        if not theory_str or not isinstance(theory_str, str):
            z = np.zeros(self.dim)
            return z.copy(), z.copy(), z.copy()
        parsed = self.parse_theory_string(theory_str)
        sprite_vectors = {}
        for class_name, class_info in parsed["classes"].items():
            emb = np.zeros(self.dim)
            emb += self._bind("name", class_name)
            if "color" in class_info:
                emb += self._bind("color", class_info["color"])
            if "vgdl_type" in class_info:
                emb += self._bind("type", class_info["vgdl_type"])
            sprite_vectors[class_name] = self._normalize(emb)
        sprite_set_vec = np.zeros(self.dim)
        for sv in sprite_vectors.values():
            sprite_set_vec += sv
        sprite_set_vec = (
            self._normalize(sprite_set_vec) if sprite_vectors else sprite_set_vec
        )
        interaction_set_vec = np.zeros(self.dim)
        for inter in parsed["interactions"]:
            emb = np.zeros(self.dim)
            emb += self._bind("effect", inter["effect"])
            agent = inter["agent"]
            emb += (
                self._bind_with_vector("agent", sprite_vectors[agent])
                if agent in sprite_vectors
                else self._bind("agent", agent)
            )
            patient = inter["patient"]
            emb += (
                self._bind_with_vector("patient", sprite_vectors[patient])
                if patient in sprite_vectors
                else self._bind("patient", patient)
            )
            emb += self._bind(
                "generic",
                "generic_true" if inter.get("generic", True) else "generic_false",
            )
            interaction_set_vec += self._normalize(emb)
        interaction_set_vec = (
            self._normalize(interaction_set_vec)
            if parsed["interactions"]
            else interaction_set_vec
        )
        termination_set_vec = np.zeros(self.dim)
        for term in parsed["terminations"]:
            emb = np.zeros(self.dim)
            emb += self._bind("rule_type", term["rule_type"])
            for key in ["sprite", "sprite1"]:
                if key in term:
                    s = term[key]
                    emb += (
                        self._bind_with_vector("sprite1", sprite_vectors[s])
                        if s in sprite_vectors
                        else self._bind("sprite1", s)
                    )
            if "sprite2" in term:
                s = term["sprite2"]
                emb += (
                    self._bind_with_vector("sprite2", sprite_vectors[s])
                    if s in sprite_vectors
                    else self._bind("sprite2", s)
                )
            if "outcome" in term:
                emb += self._bind("outcome", str(term["outcome"]))
            if "count" in term:
                emb += self._bind("count", f"count_{term['count']}")
            termination_set_vec += self._normalize(emb)
        termination_set_vec = (
            self._normalize(termination_set_vec)
            if parsed["terminations"]
            else termination_set_vec
        )
        return sprite_set_vec, interaction_set_vec, termination_set_vec


def encode_theory_sequence(encoder: HRREncoder, theory_strs: list) -> np.ndarray:
    return np.array([encoder.encode_theory(ts) for ts in theory_strs])


def encode_theory_sequence_decomposed(encoder: HRREncoder, theory_strs: list) -> tuple:
    """Encode a sequence of theories, returning three arrays of sub-vectors.

    Returns:
        (sprites, interactions, terminations) each of shape (n_states, dim)
    """
    sprites, interactions, terminations = [], [], []
    for ts in theory_strs:
        s, i, t = encoder.encode_theory_decomposed(ts)
        sprites.append(s)
        interactions.append(i)
        terminations.append(t)
    return np.array(sprites), np.array(interactions), np.array(terminations)
