"""Tests for generation utility helpers."""

from idlekv.utils.generation import last_token_logits_kwargs


class _SupportsLogitsToKeep:
    @staticmethod
    def forward(input_ids=None, logits_to_keep=0, **kwargs):
        return None


class _NoLogitsToKeep:
    @staticmethod
    def forward(input_ids=None, **kwargs):
        return None


def test_last_token_logits_kwargs_detects_supported_models():
    model = _SupportsLogitsToKeep()
    assert last_token_logits_kwargs(model) == {"logits_to_keep": 1}


def test_last_token_logits_kwargs_is_empty_for_older_signatures():
    model = _NoLogitsToKeep()
    assert last_token_logits_kwargs(model) == {}
