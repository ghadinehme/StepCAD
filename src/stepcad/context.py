"""Policy context tokenisation (the same at training and inference time)."""
from pc2cad.model_legacy_norm import float_to_bin, make_float_token, FLOAT_MAX_VAL
from pc2cad.dataset_legacy_cumul_norm import FLOAT_RE

PROMPT0 = "# Generate the full CadQuery script for this shape:\n"
PROMPT_T = "# Continue the CadQuery script for this shape:\n"


def to_bin(text):
    def sub(m):
        v = float(m.group())
        return m.group() if v >= FLOAT_MAX_VAL else make_float_token(float_to_bin(v))
    return FLOAT_RE.sub(sub, text)


def encode_context(tokenizer, tokens_per_pc, pc_pad_id, geom_pad_id, prefix):
    """Context ids for a state: prefix '' is t=0 (single-shot prompt)."""
    if not prefix:
        return [pc_pad_id] * tokens_per_pc + [geom_pad_id] + tokenizer.encode(PROMPT0, add_special_tokens=False)
    return ([pc_pad_id] * (2 * tokens_per_pc) + [geom_pad_id]
            + tokenizer.encode(PROMPT_T, add_special_tokens=False)
            + tokenizer.encode(to_bin(prefix), add_special_tokens=False))
