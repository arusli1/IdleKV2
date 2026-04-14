from types import SimpleNamespace

from transformers import GenerationConfig

from idlekv.utils.generation import greedy_generation_config


def test_greedy_generation_config_clears_sampling_fields():
    model = SimpleNamespace(
        generation_config=GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
        )
    )

    cleaned = greedy_generation_config(model)

    assert cleaned.do_sample is False
    assert cleaned.temperature is None
    assert cleaned.top_p is None
    assert cleaned.top_k is None

    # Original config should remain unchanged.
    assert model.generation_config.do_sample is True
    assert model.generation_config.temperature == 0.6
    assert model.generation_config.top_p == 0.9
    assert model.generation_config.top_k == 50
