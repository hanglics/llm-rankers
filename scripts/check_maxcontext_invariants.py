#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import ir_datasets  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ir_datasets"] = ModuleType("ir_datasets")

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    sys.modules["openai"] = ModuleType("openai")

try:
    import tiktoken  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tiktoken"] = ModuleType("tiktoken")

try:
    from pyserini.search.lucene import LuceneSearcher  # noqa: F401
    from pyserini.search._base import get_topics  # noqa: F401
except ModuleNotFoundError:
    pyserini_module = ModuleType("pyserini")
    pyserini_search = ModuleType("pyserini.search")
    pyserini_lucene = ModuleType("pyserini.search.lucene")
    pyserini_base = ModuleType("pyserini.search._base")
    pyserini_lucene.LuceneSearcher = object
    pyserini_base.get_topics = lambda *_args, **_kwargs: {}
    pyserini_search.lucene = pyserini_lucene
    pyserini_search._base = pyserini_base
    pyserini_module.search = pyserini_search
    sys.modules["pyserini"] = pyserini_module
    sys.modules["pyserini.search"] = pyserini_search
    sys.modules["pyserini.search.lucene"] = pyserini_lucene
    sys.modules["pyserini.search._base"] = pyserini_base

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


class FakeBPETokenizer:
    """Minimal Qwen-like BPE for MaxContext trie tests."""

    def __init__(
        self,
        *,
        merge_two_digit: bool = True,
        merge_leading_space_digit: bool = True,
        merge_trailing_comma: bool = True,
    ):
        self.merge_two_digit = merge_two_digit
        self.merge_leading_space_digit = merge_leading_space_digit
        self.merge_trailing_comma = merge_trailing_comma
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"
        self.last_encoded_text = None
        self._token_to_id = {"<pad>": self.pad_token_id, "<eos>": self.eos_token_id}
        self._id_to_token = {self.pad_token_id: "<pad>", self.eos_token_id: "<eos>"}

    def _token_id(self, token: str) -> int:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            token_id = len(self._token_to_id)
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return token_id

    def _tokenize_text(self, text: str) -> list[str]:
        text = str(text)
        tokens: list[str] = []
        idx = 0
        while idx < len(text):
            trailing_comma = None
            if self.merge_trailing_comma:
                trailing_comma = re.match(r" \d{1,2},", text[idx:])
            if trailing_comma:
                number = trailing_comma.group(0)[1:-1]
                if len(number) == 1 or self.merge_two_digit:
                    token = trailing_comma.group(0)
                    tokens.append(token)
                    idx += len(token)
                    continue

            if (
                self.merge_leading_space_digit
                and idx + 1 < len(text)
                and text[idx] == " "
                and text[idx + 1].isdigit()
            ):
                if idx + 2 < len(text) and text[idx + 2].isdigit() and self.merge_two_digit:
                    tokens.append(text[idx:idx + 3])
                    idx += 3
                    continue
                tokens.append(text[idx:idx + 2])
                idx += 2
                continue

            if (
                self.merge_two_digit
                and idx + 1 < len(text)
                and text[idx].isdigit()
                and text[idx + 1].isdigit()
            ):
                tokens.append(text[idx:idx + 2])
                idx += 2
                continue

            tokens.append(text[idx])
            idx += 1

        return tokens

    def _encode_text(self, text, add_special_tokens=True, record=False):
        if record:
            self.last_encoded_text = str(text)
        token_ids = [self._token_id(token) for token in self._tokenize_text(text)]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def encode(self, text, add_special_tokens=True):
        return self._encode_text(text, add_special_tokens=add_special_tokens, record=True)

    def decode(self, token_ids, skip_special_tokens=False):
        parts = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            parts.append(self._id_to_token[token_id])
        return "".join(parts)

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
        token_lists = [
            self._encode_text(text, add_special_tokens=add_special_tokens, record=False)
            for text in texts
        ]

        if truncation and max_length is not None:
            token_lists = [tokens[:max_length] for tokens in token_lists]

        if return_tensors == "pt":
            max_len = max(len(tokens) for tokens in token_lists)
            padded_ids = []
            padded_masks = []
            for tokens in token_lists:
                pad_len = max_len - len(tokens)
                padded_ids.append(tokens + ([self.pad_token_id] * pad_len))
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


def test_dual_prompt_hook_invariants():
    dual_ranker = object.__new__(DualEndSetwiseLlmRanker)
    assert (
        dual_ranker._build_dual_prompt_footer(4)
        == '\n\nOutput only in the format: Best: [label], Worst: [label]'
    )
    assert dual_ranker._get_dual_prefix_allowed_tokens_fn(4, "prompt", 100) is None

    tokenizer = FakeBPETokenizer()
    max_ranker = build_maxcontext_stub(pool_size=40, tokenizer=tokenizer)
    footer = max_ranker._build_dual_prompt_footer(40)
    assert "between 1 and 40 inclusive" in footer
    assert "X != Y" in footer

    query = "query terms"
    docs = make_docs(40)
    expected_prompt = render_dual_runtime_prompt(max_ranker, query, docs)
    expected_ids = tokenizer.encode(expected_prompt, add_special_tokens=True)
    fits, rendered_length, budget = compute_max_fit_window(
        max_ranker,
        query,
        docs,
        reserved_output_tokens=128,
    )
    assert fits is True
    assert tokenizer.last_encoded_text == expected_prompt
    assert rendered_length == len(expected_ids)
    assert budget == max_ranker.max_input_tokens - 128


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


def build_maxcontext_stub(*, pool_size: int, tokenizer=None):
    ranker = object.__new__(MaxContextDualEndSetwiseLlmRanker)
    ranker.CHARACTERS = [str(i + 1) for i in range(pool_size)]
    ranker.tokenizer = tokenizer or FakeBPETokenizer()
    ranker.llm = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=ranker.tokenizer.eos_token_id)
    )
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.device = "cpu"
    ranker.max_input_tokens = 200000
    ranker.strict_no_parse_fallback = True
    ranker.strict_no_truncation = True
    ranker.label_scheme = "numeric_1_based"
    ranker.scoring = "generation"
    ranker.num_permutation = 1
    ranker.k = pool_size
    ranker.method = "selection"
    ranker.num_child = pool_size - 1
    ranker._maxcontext_pool_size = pool_size
    ranker._comparison_log_path = None
    ranker._current_qid = None
    ranker._warned_input_truncation = False
    ranker._build_chat_prompt = lambda messages: f"<chat>{messages[0]['content']}</chat>"
    ranker._is_supported_causal_model = lambda: True
    return ranker


def make_docs(count: int):
    return [
        SimpleNamespace(docid=f"d{i + 1}", text=f"doc {i + 1}")
        for i in range(count)
    ]


def render_dual_runtime_prompt(ranker, query: str, docs):
    passages = ranker._format_passages(docs)
    input_text = (
        f'Given a query "{query}", which of the following passages is the most relevant '
        f'and which is the least relevant to the query?\n\n'
        + passages
        + ranker._build_dual_prompt_footer(len(docs))
    )
    return ranker._build_chat_prompt([{"role": "user", "content": input_text}])


def compute_continuation_ids(ranker, prompt: str, continuation: str):
    raw_kwargs = ranker._raw_tokenizer_kwargs()
    prompt_ids = ranker.tokenizer(prompt, **raw_kwargs)["input_ids"]
    full_ids = ranker.tokenizer(prompt + continuation, **raw_kwargs)["input_ids"]
    prefix_len = ranker._longest_common_prefix_length(prompt_ids, full_ids)
    continuation_ids = list(full_ids[prefix_len:])
    sentinel_ids = {ranker._resolve_dual_eos_token_id()}
    if getattr(ranker.tokenizer, "pad_token_id", None) is not None:
        sentinel_ids.add(int(ranker.tokenizer.pad_token_id))
    while continuation_ids and continuation_ids[-1] in sentinel_ids:
        continuation_ids.pop()
    return tuple(int(token_id) for token_id in continuation_ids)


def assert_reachable_prefixes(ranker, prompt_ids, allowed_fn, continuation_ids):
    for prefix_len in range(len(continuation_ids) + 1):
        current_ids = torch.tensor(
            prompt_ids + list(continuation_ids[:prefix_len]),
            dtype=torch.long,
        )
        allowed = allowed_fn(0, current_ids)
        assert allowed, f"Expected non-empty allowed set for prefix length {prefix_len}"


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


def test_dual_prefix_trie_invariants():
    test_cases = [
        (
            2,
            FakeBPETokenizer(
                merge_two_digit=True,
                merge_leading_space_digit=True,
                merge_trailing_comma=True,
            ),
            [(1, 2), (2, 1)],
        ),
        (
            10,
            FakeBPETokenizer(
                merge_two_digit=True,
                merge_leading_space_digit=False,
                merge_trailing_comma=False,
            ),
            [(9, 10), (10, 9)],
        ),
        (
            40,
            FakeBPETokenizer(
                merge_two_digit=True,
                merge_leading_space_digit=True,
                merge_trailing_comma=True,
            ),
            [(1, 40), (25, 1), (40, 33)],
        ),
    ]

    for n_docs, tokenizer, sample_pairs in test_cases:
        ranker = build_maxcontext_stub(pool_size=n_docs, tokenizer=tokenizer)
        docs = make_docs(n_docs)
        prompt = render_dual_runtime_prompt(ranker, "query", docs)
        prompt_ids = ranker.tokenizer(prompt, **ranker._raw_tokenizer_kwargs())["input_ids"]
        trie_root, eos_token_id, normalized_paths = ranker._build_dual_prefix_trie_artifacts(
            n_docs=n_docs,
            prompt=prompt,
            prompt_length=len(prompt_ids),
        )
        assert len(normalized_paths) == n_docs * (n_docs - 1)

        allowed_fn = ranker._get_dual_prefix_allowed_tokens_fn(
            n_docs=n_docs,
            prompt=prompt,
            prompt_length=len(prompt_ids),
        )
        for pair in sample_pairs:
            assert pair in normalized_paths
            assert_reachable_prefixes(ranker, prompt_ids, allowed_fn, normalized_paths[pair])

        if n_docs == 2:
            assert set(normalized_paths.keys()) == {(1, 2), (2, 1)}

        terminal_leaf_count = 0
        stack = [trie_root]
        while stack:
            node = stack.pop()
            if not node.children:
                continue
            child_keys = set(node.children.keys())
            if child_keys == {eos_token_id}:
                terminal_leaf_count += 1
                continue
            assert node.children
            assert eos_token_id not in child_keys
            stack.extend(node.children.values())

        assert terminal_leaf_count == n_docs * (n_docs - 1)


def test_dual_prefix_invalid_label_guards():
    tokenizer = FakeBPETokenizer(
        merge_two_digit=True,
        merge_leading_space_digit=True,
        merge_trailing_comma=True,
    )
    ranker = build_maxcontext_stub(pool_size=40, tokenizer=tokenizer)
    docs = make_docs(40)
    prompt = render_dual_runtime_prompt(ranker, "query", docs)
    prompt_ids = ranker.tokenizer(prompt, **ranker._raw_tokenizer_kwargs())["input_ids"]
    allowed_fn = ranker._get_dual_prefix_allowed_tokens_fn(
        n_docs=40,
        prompt=prompt,
        prompt_length=len(prompt_ids),
    )

    best_slot_prefix = compute_continuation_ids(ranker, prompt, " Best:")
    allowed_at_best = allowed_fn(
        0,
        torch.tensor(prompt_ids + list(best_slot_prefix), dtype=torch.long),
    )
    for invalid_label in range(41, 45):
        invalid_best_ids = compute_continuation_ids(ranker, prompt, f" Best: {invalid_label}")
        invalid_best_token = invalid_best_ids[len(best_slot_prefix)]
        assert invalid_best_token not in allowed_at_best

    for best_label in (1, 10, 40):
        worst_slot_prefix = compute_continuation_ids(
            ranker,
            prompt,
            f" Best: {best_label}, Worst:",
        )
        allowed_at_worst = allowed_fn(
            0,
            torch.tensor(prompt_ids + list(worst_slot_prefix), dtype=torch.long),
        )

        duplicate_worst_ids = compute_continuation_ids(
            ranker,
            prompt,
            f" Best: {best_label}, Worst: {best_label}",
        )
        duplicate_worst_token = duplicate_worst_ids[len(worst_slot_prefix)]
        assert duplicate_worst_token not in allowed_at_worst

        invalid_worst_ids = compute_continuation_ids(
            ranker,
            prompt,
            f" Best: {best_label}, Worst: 44",
        )
        invalid_worst_token = invalid_worst_ids[len(worst_slot_prefix)]
        assert invalid_worst_token not in allowed_at_worst


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
    ranker._generate = lambda inputs, max_new_tokens, prefix_allowed_tokens_fn=None: torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )
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


def test_generate_prefix_identity():
    class RecordingLLM:
        def __init__(self):
            self.calls = []
            self.generation_config = SimpleNamespace(
                do_sample=True,
                temperature=0.7,
                top_k=20,
                top_p=0.9,
                min_p=0.1,
                typical_p=0.95,
                epsilon_cutoff=0.0,
                eta_cutoff=0.0,
                pad_token_id=None,
                eos_token_id=None,
            )

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[7, 8, 9]], dtype=torch.long)

    ranker = object.__new__(SetwiseLlmRanker)
    ranker.llm = RecordingLLM()
    ranker.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    ranker._is_supported_causal_model = lambda: True

    model_inputs = DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )

    without_kwarg = ranker._generate(model_inputs, max_new_tokens=4)
    with_none_kwarg = ranker._generate(
        model_inputs,
        max_new_tokens=4,
        prefix_allowed_tokens_fn=None,
    )

    assert torch.equal(without_kwarg, with_none_kwarg)
    assert len(ranker.llm.calls) == 2
    assert all("prefix_allowed_tokens_fn" not in call for call in ranker.llm.calls)


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
    test_dual_prompt_hook_invariants()
    test_parse_invariants()
    test_dual_prefix_trie_invariants()
    test_dual_prefix_invalid_label_guards()
    test_compare_both_duplicate_rewrite_guard()
    test_tokenize_invariants()
    test_generate_prefix_identity()
    test_default_false_flags_and_logging()
    test_likelihood_substitute_and_fit_helper()
    # Authoritative plan assertion 11 (Phase-1 sanity at pool_size=40 on DL19)
    # is intentionally skipped here. It belongs to the post-implementation
    # cluster audit, not this offline invariant script.
    print("check_maxcontext_invariants.py: all checks passed")


if __name__ == "__main__":
    main()
