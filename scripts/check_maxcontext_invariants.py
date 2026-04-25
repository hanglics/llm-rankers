#!/usr/bin/env python3
from __future__ import annotations

import io
import json
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


def _missing_dependency(*_args, **_kwargs):
    raise RuntimeError("Optional runtime dependency is not available in this test environment.")


try:
    import ir_datasets  # noqa: F401
except ModuleNotFoundError:
    ir_datasets_stub = ModuleType("ir_datasets")
    ir_datasets_stub.load = _missing_dependency
    sys.modules["ir_datasets"] = ir_datasets_stub

try:
    from pyserini.search.lucene import LuceneSearcher as _LuceneSearcher  # noqa: F401
    from pyserini.search._base import get_topics as _get_topics  # noqa: F401
except ModuleNotFoundError:
    pyserini_stub = ModuleType("pyserini")
    pyserini_search_stub = ModuleType("pyserini.search")
    pyserini_lucene_stub = ModuleType("pyserini.search.lucene")
    pyserini_base_stub = ModuleType("pyserini.search._base")

    class _LuceneSearcherStub:
        from_prebuilt_index = staticmethod(_missing_dependency)

    pyserini_lucene_stub.LuceneSearcher = _LuceneSearcherStub
    pyserini_base_stub.get_topics = _missing_dependency
    sys.modules["pyserini"] = pyserini_stub
    sys.modules["pyserini.search"] = pyserini_search_stub
    sys.modules["pyserini.search.lucene"] = pyserini_lucene_stub
    sys.modules["pyserini.search._base"] = pyserini_base_stub

try:
    import tiktoken  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tiktoken"] = ModuleType("tiktoken")

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    sys.modules["openai"] = ModuleType("openai")

import run as run_module
from llmrankers.rankers import SearchResult
from llmrankers.setwise import SetwiseLlmRanker, compute_max_fit_window
from llmrankers.setwise_extended import (
    BiasAwareDualEndSetwiseLlmRanker,
    BottomUpSetwiseLlmRanker,
    DualEndSetwiseLlmRanker,
    MaxContextBottomUpSetwiseLlmRanker,
    MaxContextDualEndSetwiseLlmRanker,
    MaxContextTopDownSetwiseLlmRanker,
    SameCallRegularizedSetwiseLlmRanker,
    SelectiveDualEndSetwiseLlmRanker,
    _assert_maxcontext_bottomup_fits,
    _assert_maxcontext_topdown_fits,
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


class DummyCausalModel:
    def eval(self):
        return self


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


def make_run_args(
    *,
    hits=10,
    scoring="generation",
    num_permutation=1,
    method="selection",
    k=10,
    direction="maxcontext_dualend",
    openai_key=None,
):
    return SimpleNamespace(
        run=SimpleNamespace(
            model_name_or_path="Qwen/Qwen3-4B",
            tokenizer_name_or_path=None,
            device="cpu",
            cache_dir=None,
            openai_key=openai_key,
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
            direction=direction,
            num_child=3,
            method=method,
            k=k,
            num_permutation=num_permutation,
            character_scheme="letters_a_w",
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


def fake_super_init_factory(model_type, *, max_input_tokens=32768, tokenizer=None):
    def fake_super_init(self, *args, **kwargs):
        self.config = SimpleNamespace(model_type=model_type)
        self.scoring = kwargs["scoring"]
        self.k = kwargs["k"]
        self.num_permutation = kwargs["num_permutation"]
        self.method = kwargs["method"]
        self.max_input_tokens = max_input_tokens
        self.tokenizer = DummyTokenizer() if tokenizer is None else tokenizer
        self._comparison_log_path = None
        self._current_qid = None

    return fake_super_init


def instantiate_maxcontext_variant(
    ranker_cls,
    init_owner,
    *,
    model_type="qwen3",
    scoring="generation",
    pool_size=10,
    method="selection",
    num_permutation=1,
    k=None,
):
    if k is None:
        k = pool_size
    with mock.patch.object(init_owner, "__init__", new=fake_super_init_factory(model_type)):
        return ranker_cls(
            model_name_or_path="Qwen/Qwen3-4B",
            tokenizer_name_or_path=None,
            device="cpu",
            cache_dir=None,
            num_child=3,
            scoring=scoring,
            method=method,
            num_permutation=num_permutation,
            k=k,
            pool_size=pool_size,
        )


def make_docs(count):
    return [
        SearchResult(docid=f"d{i}", score=float(-i), text=f"doc {i}")
        for i in range(1, count + 1)
    ]


def make_docs_with_scores(scores):
    return [
        SearchResult(docid=f"d{i}", score=score, text=f"doc {i}")
        for i, score in enumerate(scores, start=1)
    ]


def assert_materialized_rerank(results, docs):
    assert len(results) == len(docs)
    assert [doc.score for doc in results] == [-rank for rank in range(1, len(docs) + 1)]
    assert all(doc.text is None for doc in results)
    output_docids = [doc.docid for doc in results]
    input_docids = [doc.docid for doc in docs]
    assert set(output_docids) == set(input_docids)
    assert len(output_docids) == len(set(output_docids))


def build_strict_topdown_compare_stub():
    ranker = object.__new__(MaxContextTopDownSetwiseLlmRanker)
    ranker.total_compare = 0
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "garbage")
    ranker.CHARACTERS = ["1", "2", "3"]
    ranker.strict_no_parse_fallback = True
    ranker._build_best_prompt = lambda query, docs: "prompt"
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._parse_single_label = lambda raw, characters: None
    ranker._is_supported_causal_model = lambda: True
    return ranker


def build_strict_bottomup_compare_stub():
    ranker = object.__new__(MaxContextBottomUpSetwiseLlmRanker)
    ranker.total_compare = 0
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "garbage")
    ranker.CHARACTERS = ["1", "2", "3"]
    ranker.strict_no_parse_fallback = True
    ranker._build_worst_prompt = lambda query, docs: "prompt"
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._parse_single_label = lambda raw, characters: None
    ranker._is_supported_causal_model = lambda: True
    return ranker


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


def test_maxcontext_dualend_byte_identity_snapshot():
    tokenizer_sentinel = "tokenizer-sentinel"
    with mock.patch.object(
        DualEndSetwiseLlmRanker,
        "__init__",
        new=fake_super_init_factory("qwen3", tokenizer=tokenizer_sentinel),
    ):
        ranker = MaxContextDualEndSetwiseLlmRanker(
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

    expected_snapshot = {
        "config": SimpleNamespace(model_type="qwen3"),
        "scoring": "generation",
        "k": 10,
        "num_permutation": 1,
        "method": "selection",
        "max_input_tokens": 32768,
        "tokenizer": tokenizer_sentinel,
        "_comparison_log_path": None,
        "_current_qid": None,
        "CHARACTERS": [str(i + 1) for i in range(10)],
        "num_child": 9,
        "strict_no_truncation": True,
        "strict_no_parse_fallback": True,
        "label_scheme": "numeric_1_based",
        "_maxcontext_pool_size": 10,
    }
    assert vars(ranker) == expected_snapshot


def test_maxcontext_topdown_invariants():
    ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=10,
    )
    assert ranker.CHARACTERS == [str(i + 1) for i in range(10)]
    assert ranker.num_child == 9
    assert ranker.method == "selection"
    assert ranker.strict_no_truncation is True
    assert ranker.strict_no_parse_fallback is True
    assert ranker.label_scheme == "numeric_1_based"
    assert ranker.total_bm25_bypass == 0

    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker,
            SetwiseLlmRanker,
            model_type="t5",
        ),
        ValueError,
        "Qwen3 / Qwen3.5 model_type",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker,
            SetwiseLlmRanker,
            scoring="likelihood",
        ),
        ValueError,
        "--scoring generation",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker,
            SetwiseLlmRanker,
            method="heapsort",
        ),
        ValueError,
        "method='selection'",
    )
    expect_raises(
        lambda: run_module.main(
            make_run_args(direction="maxcontext_topdown", openai_key="sk-test")
        ),
        ValueError,
        "--direction maxcontext_topdown is not supported with --openai_key",
    )

    docs = make_docs(10)
    ranker._assert_maxcontext_fits = lambda query, ranking: None
    call_counter = {"count": 0}

    def choose_first(query, window):
        ranker.total_compare += 1
        call_counter["count"] += 1
        return "1"

    ranker.compare = choose_first
    results = ranker.rerank("query", docs)
    assert call_counter["count"] == 8
    assert ranker.total_compare == 8
    assert ranker.total_bm25_bypass == 1
    assert_materialized_rerank(results, docs)

    one_doc_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=1,
    )
    one_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    one_doc_calls = {"count": 0}
    one_doc_ranker.compare = lambda query, window: one_doc_calls.__setitem__("count", one_doc_calls["count"] + 1) or "1"
    one_doc_results = one_doc_ranker.rerank("query", make_docs(1))
    assert one_doc_calls["count"] == 0
    assert one_doc_ranker.total_compare == 0
    assert one_doc_ranker.total_bm25_bypass == 0
    assert_materialized_rerank(one_doc_results, make_docs(1))

    two_doc_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=2,
    )
    two_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    two_doc_ranker.compare = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LLM should not be called at n_docs=2")
    )
    two_doc_results = two_doc_ranker.rerank("query", make_docs_with_scores([0.0, 1.0]))
    assert two_doc_ranker.total_compare == 0
    assert two_doc_ranker.total_bm25_bypass == 1
    assert [doc.docid for doc in two_doc_results] == ["d2", "d1"]

    two_doc_tie_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=2,
    )
    two_doc_tie_ranker._assert_maxcontext_fits = lambda query, ranking: None
    two_doc_tie_ranker.compare = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LLM should not be called at n_docs=2")
    )
    two_doc_tie_results = two_doc_tie_ranker.rerank(
        "query",
        [
            SearchResult(docid="d2", score=1.0, text="doc 2"),
            SearchResult(docid="d1", score=1.0, text="doc 1"),
        ],
    )
    assert two_doc_tie_ranker.total_compare == 0
    assert two_doc_tie_ranker.total_bm25_bypass == 1
    assert [doc.docid for doc in two_doc_tie_results] == ["d2", "d1"]

    three_doc_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=3,
    )
    three_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    three_doc_calls = {"count": 0}

    def choose_first_three(query, window):
        three_doc_ranker.total_compare += 1
        three_doc_calls["count"] += 1
        return "1"

    three_doc_ranker.compare = choose_first_three
    three_doc_results = three_doc_ranker.rerank("query", make_docs(3))
    assert three_doc_calls["count"] == 1
    assert three_doc_ranker.total_compare == 1
    assert three_doc_ranker.total_bm25_bypass == 1
    assert_materialized_rerank(three_doc_results, make_docs(3))

    bad_score_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=2,
    )
    bad_score_ranker._assert_maxcontext_fits = lambda query, ranking: None
    expect_raises(
        lambda: bad_score_ranker.rerank(
            "query",
            [
                SearchResult(docid="d1", score=None, text="doc 1"),
                SearchResult(docid="d2", score=1.0, text="doc 2"),
            ],
        ),
        ValueError,
        "requires finite BM25 scores",
    )

    strict_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=4,
    )
    strict_ranker._assert_maxcontext_fits = lambda query, ranking: None
    labels = iter(["1", "4"])
    strict_ranker.compare = lambda query, window: next(labels)
    expect_raises(
        lambda: strict_ranker.rerank("query", make_docs(4)),
        ValueError,
        "outside the active window",
    )

    compare_ranker = build_strict_topdown_compare_stub()
    expect_raises(
        lambda: compare_ranker.compare("query", make_docs(2)),
        ValueError,
        "Raw text: 'garbage'",
    )

    prompt_ranker = SimpleNamespace(
        max_input_tokens=1000,
        tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=True: list(range(10))),
        _build_best_prompt=mock.Mock(return_value="best prompt"),
        _build_chat_prompt=mock.Mock(side_effect=lambda messages: messages[0]["content"] + " chat"),
    )
    _assert_maxcontext_topdown_fits(prompt_ranker, "query", make_docs(2))
    prompt_ranker._build_best_prompt.assert_called_once_with("query", make_docs(2))


def test_maxcontext_bottomup_invariants():
    ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=10,
    )
    assert ranker.CHARACTERS == [str(i + 1) for i in range(10)]
    assert ranker.num_child == 9
    assert ranker.method == "selection"
    assert ranker.strict_no_truncation is True
    assert ranker.strict_no_parse_fallback is True
    assert ranker.label_scheme == "numeric_1_based"
    assert ranker.total_bm25_bypass == 0

    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextBottomUpSetwiseLlmRanker,
            BottomUpSetwiseLlmRanker,
            model_type="t5",
        ),
        ValueError,
        "Qwen3 / Qwen3.5 model_type",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextBottomUpSetwiseLlmRanker,
            BottomUpSetwiseLlmRanker,
            scoring="likelihood",
        ),
        ValueError,
        "--scoring generation",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextBottomUpSetwiseLlmRanker,
            BottomUpSetwiseLlmRanker,
            method="heapsort",
        ),
        ValueError,
        "method='selection'",
    )
    expect_raises(
        lambda: run_module.main(
            make_run_args(direction="maxcontext_bottomup", openai_key="sk-test")
        ),
        ValueError,
        "--direction maxcontext_bottomup is not supported with --openai_key",
    )

    docs = make_docs(10)
    ranker._assert_maxcontext_fits = lambda query, ranking: None
    call_counter = {"count": 0}

    def choose_first(query, window):
        ranker.total_compare += 1
        call_counter["count"] += 1
        return "1"

    ranker.compare_worst = choose_first
    results = ranker.rerank("query", docs)
    assert call_counter["count"] == 8
    assert ranker.total_compare == 8
    assert ranker.total_bm25_bypass == 1
    assert_materialized_rerank(results, docs)

    one_doc_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=1,
    )
    one_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    one_doc_calls = {"count": 0}
    one_doc_ranker.compare_worst = lambda query, window: one_doc_calls.__setitem__("count", one_doc_calls["count"] + 1) or "1"
    one_doc_results = one_doc_ranker.rerank("query", make_docs(1))
    assert one_doc_calls["count"] == 0
    assert one_doc_ranker.total_compare == 0
    assert one_doc_ranker.total_bm25_bypass == 0
    assert_materialized_rerank(one_doc_results, make_docs(1))

    two_doc_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=2,
    )
    two_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    two_doc_ranker.compare_worst = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LLM should not be called at n_docs=2")
    )
    two_doc_results = two_doc_ranker.rerank("query", make_docs_with_scores([0.0, 1.0]))
    assert two_doc_ranker.total_compare == 0
    assert two_doc_ranker.total_bm25_bypass == 1
    assert [doc.docid for doc in two_doc_results] == ["d2", "d1"]

    two_doc_tie_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=2,
    )
    two_doc_tie_ranker._assert_maxcontext_fits = lambda query, ranking: None
    two_doc_tie_ranker.compare_worst = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LLM should not be called at n_docs=2")
    )
    two_doc_tie_results = two_doc_tie_ranker.rerank(
        "query",
        [
            SearchResult(docid="d2", score=1.0, text="doc 2"),
            SearchResult(docid="d1", score=1.0, text="doc 1"),
        ],
    )
    assert two_doc_tie_ranker.total_compare == 0
    assert two_doc_tie_ranker.total_bm25_bypass == 1
    assert [doc.docid for doc in two_doc_tie_results] == ["d2", "d1"]

    three_doc_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=3,
    )
    three_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    three_doc_calls = {"count": 0}

    def choose_first_three(query, window):
        three_doc_ranker.total_compare += 1
        three_doc_calls["count"] += 1
        return "1"

    three_doc_ranker.compare_worst = choose_first_three
    three_doc_results = three_doc_ranker.rerank("query", make_docs(3))
    assert three_doc_calls["count"] == 1
    assert three_doc_ranker.total_compare == 1
    assert three_doc_ranker.total_bm25_bypass == 1
    assert_materialized_rerank(three_doc_results, make_docs(3))

    strict_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=4,
    )
    strict_ranker._assert_maxcontext_fits = lambda query, ranking: None
    labels = iter(["1", "4"])
    strict_ranker.compare_worst = lambda query, window: next(labels)
    expect_raises(
        lambda: strict_ranker.rerank("query", make_docs(4)),
        ValueError,
        "outside the active window",
    )

    bad_score_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=2,
    )
    bad_score_ranker._assert_maxcontext_fits = lambda query, ranking: None
    expect_raises(
        lambda: bad_score_ranker.rerank(
            "query",
            [
                SearchResult(docid="d1", score=None, text="doc 1"),
                SearchResult(docid="d2", score=1.0, text="doc 2"),
            ],
        ),
        ValueError,
        "requires finite BM25 scores",
    )

    compare_ranker = build_strict_bottomup_compare_stub()
    expect_raises(
        lambda: compare_ranker.compare_worst("query", make_docs(2)),
        ValueError,
        "Raw text: 'garbage'",
    )

    prompt_ranker = SimpleNamespace(
        max_input_tokens=1000,
        tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=True: list(range(10))),
        _build_worst_prompt=mock.Mock(return_value="worst prompt"),
        _build_chat_prompt=mock.Mock(side_effect=lambda messages: messages[0]["content"] + " chat"),
    )
    _assert_maxcontext_bottomup_fits(prompt_ranker, "query", make_docs(2))
    prompt_ranker._build_worst_prompt.assert_called_once_with("query", make_docs(2))


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


def test_single_label_parser_hardening():
    numeric_valid = [str(i) for i in range(1, 31)]
    numeric_ranker = build_single_label_parser_stub(
        strict=True,
        label_scheme="numeric_1_based",
        characters=numeric_valid,
    )
    assert numeric_ranker._parse_single_label("10", numeric_valid) == "10"
    assert numeric_ranker._parse_single_label("22", numeric_valid) == "22"
    assert numeric_ranker._parse_single_label("30", numeric_valid) == "30"
    assert (
        numeric_ranker._parse_single_label(
            "Passage 3 is most relevant, the others are equally relevant",
            numeric_valid[:10],
        )
        == "3"
    )
    assert numeric_ranker._parse_single_label(
        "there is no least relevant",
        numeric_valid[:10],
    ) is None
    assert numeric_ranker._parse_single_label(
        "If there was a Passage 3, it might be relevant",
        numeric_valid[:10],
    ) is None

    letter_valid = [chr(ord("A") + i) for i in range(23)]
    legacy_letter_ranker = build_single_label_parser_stub(characters=letter_valid)
    assert legacy_letter_ranker._parse_single_label("10", letter_valid) == "J"
    assert legacy_letter_ranker._parse_single_label("A", ["A", "B", "C"]) == "A"


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


def build_single_label_parser_stub(*, strict=False, label_scheme=None, characters=None):
    ranker = object.__new__(SetwiseLlmRanker)
    ranker.CHARACTERS = characters or [chr(ord("A") + i) for i in range(23)]
    ranker.strict_no_parse_fallback = strict
    if label_scheme is not None:
        ranker.label_scheme = label_scheme
    return ranker


def make_stub_causal_tokenizer():
    return SimpleNamespace(pad_token=None, eos_token="</s>", model_max_length=4096)


def instantiate_setwise_ranker(
    ranker_cls=SetwiseLlmRanker,
    *,
    model_type="qwen3",
    scoring="generation",
    character_scheme="letters_a_w",
):
    config = SimpleNamespace(model_type=model_type, n_positions=4096, max_position_embeddings=4096)
    with mock.patch("llmrankers.setwise.AutoConfig.from_pretrained", return_value=config), \
         mock.patch("llmrankers.setwise.AutoTokenizer.from_pretrained", return_value=make_stub_causal_tokenizer()), \
         mock.patch("llmrankers.setwise.AutoModelForCausalLM.from_pretrained", return_value=DummyCausalModel()):
        return ranker_cls(
            model_name_or_path="Qwen/Qwen3-4B",
            tokenizer_name_or_path=None,
            device="cpu",
            num_child=3,
            k=10,
            scoring=scoring,
            character_scheme=character_scheme,
            method="heapsort",
            num_permutation=1,
            cache_dir=None,
        )


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


def test_topdown_bigram_scheme_invariants():
    default_ranker = instantiate_setwise_ranker()
    assert not hasattr(default_ranker, "label_scheme")
    assert default_ranker.CHARACTERS == [chr(ord("A") + i) for i in range(23)]

    bigram_ranker = instantiate_setwise_ranker(character_scheme="bigrams_aa_zz")
    assert len(bigram_ranker.CHARACTERS) == 676
    assert bigram_ranker.CHARACTERS[0] == "AA"
    assert bigram_ranker.CHARACTERS[-1] == "ZZ"
    assert bigram_ranker.label_scheme == "bigrams_aa_zz"

    with mock.patch(
        "llmrankers.setwise.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type="t5"),
    ):
        expect_raises(
            lambda: SetwiseLlmRanker(
                model_name_or_path="google/flan-t5-xl",
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="generation",
                character_scheme="bigrams_aa_zz",
                method="heapsort",
                num_permutation=1,
                cache_dir=None,
            ),
            ValueError,
            "T5",
        )

    with mock.patch(
        "llmrankers.setwise.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type="qwen3"),
    ):
        expect_raises(
            lambda: SetwiseLlmRanker(
                model_name_or_path="Qwen/Qwen3-4B",
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="likelihood",
                character_scheme="bigrams_aa_zz",
                method="heapsort",
                num_permutation=1,
                cache_dir=None,
            ),
            ValueError,
            "--scoring likelihood",
        )

    with mock.patch(
        "llmrankers.setwise.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type="qwen3"),
    ):
        expect_raises(
            lambda: DualEndSetwiseLlmRanker(
                model_name_or_path="Qwen/Qwen3-4B",
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="generation",
                character_scheme="bigrams_aa_zz",
                method="heapsort",
                num_permutation=1,
                cache_dir=None,
            ),
            ValueError,
            "only supported on SetwiseLlmRanker",
        )

    legacy_parser_ranker = object.__new__(SetwiseLlmRanker)
    assert not hasattr(legacy_parser_ranker, "label_scheme")
    assert legacy_parser_ranker._parse_single_label("Best: [B]", ["A", "B", "C"]) == "B"

    parser_ranker = object.__new__(SetwiseLlmRanker)
    parser_ranker.label_scheme = "bigrams_aa_zz"
    valid = ["AA", "AB", "AC"]
    assert parser_ranker._parse_single_label("Best: AB", valid) == "AB"
    assert parser_ranker._parse_single_label("[AB]", valid) == "AB"
    assert parser_ranker._parse_single_label("AB", valid) == "AB"
    assert parser_ranker._parse_single_label("I recommend AB testing", valid) is None


def main():
    test_dispatch_invariants()
    test_maxcontext_init_invariants()
    test_maxcontext_dualend_byte_identity_snapshot()
    test_maxcontext_topdown_invariants()
    test_maxcontext_bottomup_invariants()
    test_parse_invariants()
    test_single_label_parser_hardening()
    test_compare_both_duplicate_rewrite_guard()
    test_tokenize_invariants()
    test_default_false_flags_and_logging()
    test_likelihood_substitute_and_fit_helper()
    test_topdown_bigram_scheme_invariants()
    print("check_maxcontext_invariants.py: all checks passed")


if __name__ == "__main__":
    main()
