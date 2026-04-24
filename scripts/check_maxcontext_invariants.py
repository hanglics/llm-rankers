#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run as run_module
from llmrankers.setwise import SetwiseLlmRanker, compute_max_fit_window
from llmrankers.setwise_extended import (
    BiasAwareDualEndSetwiseLlmRanker,
    DualEndSetwiseLlmRanker,
    MaxContextDualEndSetwiseLlmRanker,
    SameCallRegularizedSetwiseLlmRanker,
    SelectiveDualEndSetwiseLlmRanker,
)


class DummyBatch(dict):
    def __getattr__(self, item):
        return self[item]

    def to(self, _device):
        return self


class DummyTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(self._length(text, add_special_tokens)))

    def __call__(
        self,
        inputs,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors=None,
        truncation=False,
        max_length=None,
        padding=False,
    ):
        single_input = isinstance(inputs, str)
        texts = [inputs] if single_input else list(inputs)
        token_lists = [list(range(self._length(text, add_special_tokens))) for text in texts]

        if truncation and max_length is not None:
            token_lists = [tokens[:max_length] for tokens in token_lists]

        if return_tensors == "pt":
            max_len = max(len(tokens) for tokens in token_lists)
            padded_ids = []
            padded_masks = []
            for tokens in token_lists:
                pad_len = max_len - len(tokens)
                padded_ids.append(tokens + ([0] * pad_len))
                padded_masks.append(([1] * len(tokens)) + ([0] * pad_len))
            return DummyBatch(
                {
                    "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
                }
            )

        if single_input:
            return {
                "input_ids": token_lists[0],
                "attention_mask": [1] * len(token_lists[0]),
            }
        return {
            "input_ids": token_lists,
            "attention_mask": [[1] * len(tokens) for tokens in token_lists],
        }

    @staticmethod
    def _length(text, add_special_tokens):
        base = len(str(text).split())
        return base + (1 if add_special_tokens else 0)


def expect_raises(fn, exc_type, contains: str):
    try:
        fn()
    except exc_type as exc:
        if contains and contains not in str(exc):
            raise AssertionError(f"Expected {contains!r} in {exc!r}") from exc
        return
    except Exception as exc:  # pragma: no cover - defensive
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised.")


def make_run_args(*, hits=10, scoring="generation", num_permutation=1, method="selection", k=10):
    return SimpleNamespace(
        run=SimpleNamespace(
            model_name_or_path="Qwen/Qwen3-4B",
            tokenizer_name_or_path=None,
            device="cpu",
            cache_dir=None,
            openai_key=None,
            scoring=scoring,
            hits=hits,
            query_length=128,
            passage_length=512,
            log_comparisons=None,
            ir_dataset_name=None,
            pyserini_index=None,
            run_path="unused.txt",
            save_path="unused.txt",
            shuffle_ranking=None,
        ),
        setwise=SimpleNamespace(
            direction="maxcontext_dualend",
            num_child=3,
            method=method,
            k=k,
            num_permutation=num_permutation,
            fusion="rrf",
            alpha=0.5,
            gate_strategy="hybrid",
            shortlist_size=20,
            margin_threshold=0.15,
            uncertainty_percentile=None,
            order_robust_orderings=3,
        ),
        pointwise=None,
        pairwise=None,
        listwise=None,
    )


def test_dispatch_invariants():
    expect_raises(
        lambda: run_module.main(make_run_args(hits=20, k=10)),
        ValueError,
        "--hits == --k",
    )
    expect_raises(
        lambda: run_module.main(make_run_args(scoring="likelihood")),
        ValueError,
        "--scoring generation",
    )
    expect_raises(
        lambda: run_module.main(make_run_args(num_permutation=2)),
        ValueError,
        "--num_permutation 1",
    )


def test_maxcontext_init_invariants():
    def fake_super_init_factory(model_type):
        def fake_super_init(self, *args, **kwargs):
            self.config = SimpleNamespace(model_type=model_type)
            self.scoring = kwargs["scoring"]
            self.k = kwargs["k"]
            self.num_permutation = kwargs["num_permutation"]
            self.method = kwargs["method"]
            self.max_input_tokens = 32768
            self.tokenizer = DummyTokenizer()
            self._comparison_log_path = None

        return fake_super_init

    common_kwargs = dict(
        model_name_or_path="Qwen/Qwen3-4B",
        tokenizer_name_or_path=None,
        device="cpu",
        cache_dir=None,
        num_child=3,
        scoring="generation",
        method="selection",
        num_permutation=1,
        k=10,
        pool_size=10,
    )

    with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_super_init_factory("qwen3")):
        ranker = MaxContextDualEndSetwiseLlmRanker(**common_kwargs)
    assert ranker.CHARACTERS == [str(i + 1) for i in range(10)]
    assert ranker.num_child == 9
    assert ranker.method == "selection"
    assert ranker.strict_no_truncation is True
    assert ranker.strict_no_parse_fallback is True
    assert ranker.label_scheme == "numeric_1_based"

    with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_super_init_factory("t5")):
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**common_kwargs),
            ValueError,
            "Qwen3 / Qwen3.5 model_type",
        )

    with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_super_init_factory("qwen2")):
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**common_kwargs),
            ValueError,
            "Qwen3 / Qwen3.5 model_type",
        )

    with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_super_init_factory("qwen3")):
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**{**common_kwargs, "scoring": "likelihood"}),
            ValueError,
            "--scoring generation",
        )
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**{**common_kwargs, "num_permutation": 2}),
            ValueError,
            "--num_permutation 1",
        )
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**{**common_kwargs, "method": "bubblesort"}),
            ValueError,
            "method='selection'",
        )
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**{**common_kwargs, "k": 9}),
            ValueError,
            "pool_size=10 but ranker.k=9",
        )


def build_dualend_stub(*, strict=False):
    ranker = object.__new__(DualEndSetwiseLlmRanker)
    ranker.CHARACTERS = [chr(ord("A") + i) for i in range(23)]
    ranker.strict_no_parse_fallback = strict
    ranker.strict_no_truncation = False
    ranker._comparison_log_path = None
    ranker._current_qid = None
    return ranker


def build_numeric_dualend_stub():
    ranker = object.__new__(DualEndSetwiseLlmRanker)
    ranker.CHARACTERS = [str(i + 1) for i in range(10)]
    ranker.strict_no_parse_fallback = True
    ranker.strict_no_truncation = False
    ranker.label_scheme = "numeric_1_based"
    ranker._comparison_log_path = None
    ranker._current_qid = None
    return ranker


def test_parse_invariants():
    strict_ranker = build_dualend_stub(strict=True)
    expect_raises(
        lambda: strict_ranker._parse_dual_output("Best: 9, Worst: 2", 3),
        ValueError,
        "parse failed",
    )
    expect_raises(
        lambda: strict_ranker._parse_dual_output("Best: 1, Worst: 1", 3),
        ValueError,
        "parse failed",
    )

    numeric_ranker = build_numeric_dualend_stub()
    assert numeric_ranker._try_parse_dual_output("Best: Passage 3  \nWorst: Passage 4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best: 3, Worst: 4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best: 3 and Worst: 4", 10) == ("3", "4")
    assert (
        numeric_ranker._try_parse_dual_output(
            "Most relevant: Passage 3 ... Least relevant: Passage 4",
            10,
        )
        == ("3", "4")
    )
    assert numeric_ranker._try_parse_dual_output("Best: Passage 10, Worst: Passage 2", 10) == ("10", "2")
    assert numeric_ranker._parse_dual_output("Best: Passage 3  \nWorst: Passage 4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best: 3, Worst: 3", 10) is None
    assert numeric_ranker._try_parse_dual_output("Best: 17, Worst: 42", 10) is None

    relaxed_ranker = build_dualend_stub(strict=False)
    assert relaxed_ranker._try_parse_dual_output("Best: Passage A, Worst: Passage B", 4) == ("A", "B")
    assert relaxed_ranker._parse_dual_output("Best: 1, Worst: 1", 3) == ("A", "A")
    assert relaxed_ranker._parse_dual_output("mangled output", 4) == ("A", "D")
    assert relaxed_ranker._parse_dual_output("###", 3) == ("A", "C")


def test_compare_both_duplicate_rewrite_guard():
    ranker = build_dualend_stub(strict=True)
    ranker.total_compare = 0
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "Best: 1, Worst: 1")
    ranker._format_passages = lambda docs: "\n".join(doc.text for doc in docs)
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._parse_dual_output = lambda output, n_docs: ("1", "1")

    docs = [SimpleNamespace(text="doc 1", docid="d1"), SimpleNamespace(text="doc 2", docid="d2")]
    expect_raises(
        lambda: ranker.compare_both("query", docs),
        ValueError,
        "Duplicate best/worst label",
    )


def build_setwise_tokenizer_stub(*, strict=False):
    ranker = object.__new__(SetwiseLlmRanker)
    ranker.tokenizer = DummyTokenizer()
    ranker.max_input_tokens = 5
    ranker._warned_input_truncation = False
    ranker.strict_no_truncation = strict
    ranker.device = "cpu"
    return ranker


def test_tokenize_invariants():
    long_prompt = "one two three four five six seven"

    strict_ranker = build_setwise_tokenizer_stub(strict=True)
    expect_raises(
        lambda: strict_ranker._tokenize_inputs(long_prompt),
        ValueError,
        "strict_no_truncation=True",
    )

    relaxed_ranker = build_setwise_tokenizer_stub(strict=False)
    capture = io.StringIO()
    with redirect_stdout(capture):
        batch = relaxed_ranker._tokenize_inputs(long_prompt)
    assert "Warning: prompt length" in capture.getvalue()
    assert batch.input_ids.shape[1] == relaxed_ranker.max_input_tokens


def test_default_false_flags_and_logging():
    assert getattr(object.__new__(SelectiveDualEndSetwiseLlmRanker), "strict_no_truncation") is False
    assert getattr(object.__new__(SelectiveDualEndSetwiseLlmRanker), "strict_no_parse_fallback") is False
    assert getattr(object.__new__(BiasAwareDualEndSetwiseLlmRanker), "strict_no_truncation") is False
    assert getattr(object.__new__(BiasAwareDualEndSetwiseLlmRanker), "strict_no_parse_fallback") is False
    assert getattr(object.__new__(SameCallRegularizedSetwiseLlmRanker), "strict_no_truncation") is False
    assert getattr(object.__new__(SameCallRegularizedSetwiseLlmRanker), "strict_no_parse_fallback") is False

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "comparisons.jsonl"
        ranker = object.__new__(SetwiseLlmRanker)
        ranker._comparison_log_path = str(log_path)
        ranker._current_qid = "q1"
        ranker._log_comparison("best", ["A", "B"], "A")
        entry = json.loads(log_path.read_text().strip())
        assert set(entry.keys()) == {"qid", "type", "positions", "selected"}


def test_likelihood_substitute_and_fit_helper():
    ranker = build_dualend_stub(strict=False)
    ranker.scoring = "likelihood"
    ranker.config = SimpleNamespace(model_type="qwen3")
    assert ranker._parse_dual_output("###", 3) == ("A", "C")

    helper_ranker = object.__new__(SetwiseLlmRanker)
    helper_ranker.max_input_tokens = 20
    helper_ranker.tokenizer = DummyTokenizer()
    helper_ranker._format_passages = lambda docs: "\n".join(f"Passage {i+1}: {doc.text}" for i, doc in enumerate(docs))
    helper_ranker._build_chat_prompt = lambda messages: messages[0]["content"] + " chat"
    helper_ranker._is_supported_causal_model = lambda: True
    docs = [SimpleNamespace(text="alpha beta"), SimpleNamespace(text="gamma delta")]
    fits, rendered_length, budget = compute_max_fit_window(helper_ranker, "query terms", docs, reserved_output_tokens=4)
    assert isinstance(fits, bool)
    assert rendered_length > 0
    assert budget == 16


def main():
    test_dispatch_invariants()
    test_maxcontext_init_invariants()
    test_parse_invariants()
    test_compare_both_duplicate_rewrite_guard()
    test_tokenize_invariants()
    test_default_false_flags_and_logging()
    test_likelihood_substitute_and_fit_helper()
    print("check_maxcontext_invariants.py: all checks passed")


if __name__ == "__main__":
    main()
