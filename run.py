import logging
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from llmrankers.rankers import SearchResult
from llmrankers.pointwise import PointwiseLlmRanker, MonoT5LlmRanker
from llmrankers.setwise import SetwiseLlmRanker, OpenAiSetwiseLlmRanker
from llmrankers.setwise_extended import (
    BiasAwareDualEndSetwiseLlmRanker,
    BidirectionalEnsembleRanker,
    BottomUpSetwiseLlmRanker,
    DualEndSetwiseLlmRanker,
    MaxContextBottomUpSetwiseLlmRanker,
    MaxContextDualEndSetwiseLlmRanker,
    MaxContextTopDownSetwiseLlmRanker,
    SameCallRegularizedSetwiseLlmRanker,
    SelectiveDualEndSetwiseLlmRanker,
)
from llmrankers.pairwise import PairwiseLlmRanker, DuoT5LlmRanker, OpenAiPairwiseLlmRanker
from llmrankers.listwise import OpenAiListwiseLlmRanker, ListwiseLlmRanker
from tqdm import tqdm
import argparse
import sys
import json
import time
import random
random.seed(929)
logger = logging.getLogger(__name__)


MAXCONTEXT_DIRECTIONS = {"maxcontext_dualend", "maxcontext_topdown", "maxcontext_bottomup"}


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):
    if args.setwise is not None and args.setwise.direction in MAXCONTEXT_DIRECTIONS:
        if args.run.openai_key is not None:
            raise ValueError(
                f"--direction {args.setwise.direction} is not supported with --openai_key. "
                f"MaxContext requires a local Qwen3 / Qwen3.5 model."
            )

    if args.pointwise:
        if 'monot5' in args.run.model_name_or_path:
            ranker = MonoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                     tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                     device=args.run.device,
                                     cache_dir=args.run.cache_dir,
                                     method=args.pointwise.method,
                                     batch_size=args.pointwise.batch_size)
        else:
            ranker = PointwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                        tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                        device=args.run.device,
                                        cache_dir=args.run.cache_dir,
                                        method=args.pointwise.method,
                                        batch_size=args.pointwise.batch_size)

    elif args.setwise:
        if args.run.openai_key:
            ranker = OpenAiSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                            api_key=args.run.openai_key,
                                            num_child=args.setwise.num_child,
                                            method=args.setwise.method,
                                            k=args.setwise.k)
        elif args.setwise.direction == 'topdown':
            ranker = SetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                      device=args.run.device,
                                      cache_dir=args.run.cache_dir,
                                      num_child=args.setwise.num_child,
                                      scoring=args.run.scoring,
                                      character_scheme=args.setwise.character_scheme,
                                      method=args.setwise.method,
                                      num_permutation=args.setwise.num_permutation,
                                      k=args.setwise.k)
        elif args.setwise.direction == 'bottomup':
            ranker = BottomUpSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                              tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                              device=args.run.device,
                                              cache_dir=args.run.cache_dir,
                                              num_child=args.setwise.num_child,
                                              scoring=args.run.scoring,
                                              method=args.setwise.method,
                                              num_permutation=args.setwise.num_permutation,
                                              k=args.setwise.k)
        elif args.setwise.direction == 'dualend':
            ranker = DualEndSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                             device=args.run.device,
                                             cache_dir=args.run.cache_dir,
                                             num_child=args.setwise.num_child,
                                             scoring=args.run.scoring,
                                             method=args.setwise.method,
                                             num_permutation=args.setwise.num_permutation,
                                             k=args.setwise.k)
        elif args.setwise.direction == 'selective_dualend':
            ranker = SelectiveDualEndSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                                      device=args.run.device,
                                                      cache_dir=args.run.cache_dir,
                                                      num_child=args.setwise.num_child,
                                                      scoring=args.run.scoring,
                                                      method=args.setwise.method,
                                                      num_permutation=args.setwise.num_permutation,
                                                      k=args.setwise.k,
                                                      gate_strategy=args.setwise.gate_strategy,
                                                      shortlist_size=args.setwise.shortlist_size,
                                                      margin_threshold=args.setwise.margin_threshold,
                                                      uncertainty_percentile=args.setwise.uncertainty_percentile)
        elif args.setwise.direction == 'bias_aware_dualend':
            if args.setwise.method == 'heapsort':
                raise ValueError(
                    'bias_aware_dualend supports only bubblesort and selection; '
                    'heapsort bypasses the order-robust joint prompting path.'
                )
            ranker = BiasAwareDualEndSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                                      device=args.run.device,
                                                      cache_dir=args.run.cache_dir,
                                                      num_child=args.setwise.num_child,
                                                      scoring=args.run.scoring,
                                                      method=args.setwise.method,
                                                      num_permutation=args.setwise.num_permutation,
                                                      k=args.setwise.k,
                                                      gate_strategy=args.setwise.gate_strategy,
                                                      shortlist_size=args.setwise.shortlist_size,
                                                      margin_threshold=args.setwise.margin_threshold,
                                                      uncertainty_percentile=args.setwise.uncertainty_percentile,
                                                      order_robust_orderings=args.setwise.order_robust_orderings)
        elif args.setwise.direction == 'samecall_regularized':
            ranker = SameCallRegularizedSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                                         tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                                         device=args.run.device,
                                                         cache_dir=args.run.cache_dir,
                                                         num_child=args.setwise.num_child,
                                                         scoring=args.run.scoring,
                                                         method=args.setwise.method,
                                                         num_permutation=args.setwise.num_permutation,
                                                         k=args.setwise.k)
        elif args.setwise.direction == 'bidirectional':
            ranker = BidirectionalEnsembleRanker(model_name_or_path=args.run.model_name_or_path,
                                                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                                device=args.run.device,
                                                num_child=args.setwise.num_child,
                                                k=args.setwise.k,
                                                scoring=args.run.scoring,
                                                method=args.setwise.method,
                                                num_permutation=args.setwise.num_permutation,
                                                fusion=args.setwise.fusion,
                                                alpha=args.setwise.alpha,
                                                cache_dir=args.run.cache_dir)
        elif args.setwise.direction == 'maxcontext_dualend':
            if args.run.hits != args.setwise.k:
                raise ValueError(
                    "maxcontext_dualend requires --hits == --k (pool_size)."
                )
            if args.run.scoring != "generation":
                raise ValueError(
                    "maxcontext_dualend requires --scoring generation."
                )
            if args.setwise.num_permutation != 1:
                raise ValueError(
                    "maxcontext_dualend requires --num_permutation 1."
                )
            if args.setwise.method != "selection":
                raise ValueError(
                    "maxcontext_dualend requires --method selection."
                )
            ranker = MaxContextDualEndSetwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                num_child=args.setwise.num_child,
                scoring=args.run.scoring,
                method=args.setwise.method,
                num_permutation=args.setwise.num_permutation,
                k=args.setwise.k,
                pool_size=args.setwise.k,
            )
        elif args.setwise.direction == 'maxcontext_topdown':
            if args.run.hits != args.setwise.k:
                raise ValueError(
                    "maxcontext_topdown requires --hits == --k (pool_size)."
                )
            if args.run.scoring != "generation":
                raise ValueError(
                    "maxcontext_topdown requires --scoring generation."
                )
            if args.setwise.num_permutation != 1:
                raise ValueError(
                    "maxcontext_topdown requires --num_permutation 1."
                )
            if args.setwise.method != "selection":
                raise ValueError(
                    "maxcontext_topdown requires --method selection."
                )
            ranker = MaxContextTopDownSetwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                num_child=args.setwise.num_child,
                scoring=args.run.scoring,
                method=args.setwise.method,
                num_permutation=args.setwise.num_permutation,
                k=args.setwise.k,
                pool_size=args.setwise.k,
            )
        elif args.setwise.direction == 'maxcontext_bottomup':
            if args.run.hits != args.setwise.k:
                raise ValueError(
                    "maxcontext_bottomup requires --hits == --k (pool_size)."
                )
            if args.run.scoring != "generation":
                raise ValueError(
                    "maxcontext_bottomup requires --scoring generation."
                )
            if args.setwise.num_permutation != 1:
                raise ValueError(
                    "maxcontext_bottomup requires --num_permutation 1."
                )
            if args.setwise.method != "selection":
                raise ValueError(
                    "maxcontext_bottomup requires --method selection."
                )
            ranker = MaxContextBottomUpSetwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                num_child=args.setwise.num_child,
                scoring=args.run.scoring,
                method=args.setwise.method,
                num_permutation=args.setwise.num_permutation,
                k=args.setwise.k,
                pool_size=args.setwise.k,
            )
        else:
            raise ValueError(f'Unknown direction: {args.setwise.direction}')

    elif args.pairwise:
        if args.pairwise.method != 'allpair':
            args.pairwise.batch_size = 2
            logger.info(f'Setting batch_size to 2.')

        if args.run.openai_key:
            ranker = OpenAiPairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             method=args.pairwise.method,
                                             k=args.pairwise.k)

        elif 'duot5' in args.run.model_name_or_path:
            ranker = DuoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                    tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                    device=args.run.device,
                                    cache_dir=args.run.cache_dir,
                                    method=args.pairwise.method,
                                    batch_size=args.pairwise.batch_size,
                                    k=args.pairwise.k)
        else:
            ranker = PairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       method=args.pairwise.method,
                                       batch_size=args.pairwise.batch_size,
                                       k=args.pairwise.k)

    elif args.listwise:
        if args.run.openai_key:
            ranker = OpenAiListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             window_size=args.listwise.window_size,
                                             step_size=args.listwise.step_size,
                                             num_repeat=args.listwise.num_repeat)
        else:
            ranker = ListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       window_size=args.listwise.window_size,
                                       step_size=args.listwise.step_size,
                                       scoring=args.run.scoring,
                                       num_repeat=args.listwise.num_repeat)
    else:
        raise ValueError('Must specify either --pointwise, --setwise, --pairwise or --listwise.')

    # Set up comparison logging for position bias analysis
    if args.run.log_comparisons:
        ranker._comparison_log_path = args.run.log_comparisons
        # Clear the log file
        open(args.run.log_comparisons, 'w').close()
    else:
        ranker._comparison_log_path = None

    query_map = {}
    if args.run.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        for query in dataset.queries_iter():
            qid = query.query_id
            text = query.text
            query_map[qid] = ranker.truncate(text, args.run.query_length)
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        docstore = dataset.docs_store()
    else:
        topics = get_topics(args.run.pyserini_index+'-test')
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index+'.flat')

    logger.info(f'Loading first stage run from {args.run.run_path}.')
    first_stage_rankings = []
    with open(args.run.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= args.run.hits:
                continue
            if args.run.ir_dataset_name is not None:
                text = docstore.get(docid).text
                if 'title' in dir(docstore.get(docid)):
                    text = f'{docstore.get(docid).title} {text}'
            else:
                data = json.loads(docstore.doc(docid).raw())
                text = data['text']
                if 'title' in data:
                    text = f'{data["title"]} {text}'
            text = ranker.truncate(text, args.run.passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))

    reranked_results = []
    total_comparisons = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    optional_stat_labels = {
        "total_dual_invocations": "dual invocations",
        "total_single_invocations": "single invocations",
        "total_order_robust_windows": "order-robust windows",
        "total_extra_orderings": "extra orderings",
        "total_regularized_worst_moves": "regularized worst moves",
    }
    optional_stat_totals = {
        attr: 0.0 for attr in optional_stat_labels if hasattr(ranker, attr)
    }

    tic = time.time()
    for qid, query, ranking in tqdm(first_stage_rankings):
        if args.run.shuffle_ranking is not None:
            if args.run.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif args.run.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')
        ranker._current_qid = qid
        reranked_results.append((qid, query, ranker.rerank(query, ranking)))
        total_comparisons += ranker.total_compare
        total_prompt_tokens += ranker.total_prompt_tokens
        total_completion_tokens += ranker.total_completion_tokens
        for attr in optional_stat_totals:
            optional_stat_totals[attr] += getattr(ranker, attr, 0.0)
    toc = time.time()

    print(f'Avg comparisons: {total_comparisons/len(reranked_results)}')
    print(f'Avg prompt tokens: {total_prompt_tokens/len(reranked_results)}')
    print(f'Avg completion tokens: {total_completion_tokens/len(reranked_results)}')
    print(f'Avg time per query: {(toc-tic)/len(reranked_results)}')
    for attr, total in optional_stat_totals.items():
        print(f'Avg {optional_stat_labels[attr]}: {total/len(reranked_results)}')

    write_run_file(args.run.save_path, reranked_results, 'LLMRankers')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument('--run_path', type=str, help='Path to the first stage run file (TREC format) to rerank.')
    run_parser.add_argument('--save_path', type=str, help='Path to save the reranked run file (TREC format).')
    run_parser.add_argument('--model_name_or_path', type=str,
                            help='Path to the pretrained model or model identifier from huggingface.co/models')
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                            help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    run_parser.add_argument('--ir_dataset_name', type=str, default=None)
    run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=128)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    run_parser.add_argument('--openai_key', type=str, default=None)
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    run_parser.add_argument('--log_comparisons', type=str, default=None,
                            help='Path to write per-comparison JSONL log for position bias analysis')

    pointwise_parser = commands.add_parser('pointwise')
    pointwise_parser.add_argument('--method', type=str, default='yes_no',
                                  choices=['qlm', 'yes_no'])
    pointwise_parser.add_argument('--batch_size', type=int, default=2)

    pairwise_parser = commands.add_parser('pairwise')
    pairwise_parser.add_argument('--method', type=str, default='allpair',
                                 choices=['allpair', 'heapsort', 'bubblesort'])
    pairwise_parser.add_argument('--batch_size', type=int, default=2)
    pairwise_parser.add_argument('--k', type=int, default=10)

    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--num_child', type=int, default=3)
    setwise_parser.add_argument('--method', type=str, default='heapsort',
                                choices=['heapsort', 'bubblesort', 'selection'])
    setwise_parser.add_argument('--k', type=int, default=10)
    setwise_parser.add_argument('--num_permutation', type=int, default=1)
    setwise_parser.add_argument('--direction', type=str, default='topdown',
                                choices=['topdown', 'bottomup', 'dualend', 'selective_dualend',
                                         'bias_aware_dualend', 'samecall_regularized', 'bidirectional',
                                         'maxcontext_dualend', 'maxcontext_topdown',
                                         'maxcontext_bottomup'],
                                help='Ranking direction: topdown (standard), bottomup (reverse), '
                                     'dualend (simultaneous best-worst), selective_dualend '
                                     '(TopDown with selective joint prompting), bias_aware_dualend '
                                     '(order-robust joint prompting), samecall_regularized '
                                     '(TopDown with worst-signal regularization), bidirectional (ensemble), '
                                     'maxcontext_dualend (full-pool numeric DualEnd selection), '
                                     'maxcontext_topdown (full-pool numeric best-only selection), '
                                     'maxcontext_bottomup (full-pool numeric worst-only selection)')
    setwise_parser.add_argument('--character_scheme', type=str, default='letters_a_w',
                                choices=['letters_a_w', 'bigrams_aa_zz'])
    setwise_parser.add_argument('--fusion', type=str, default='rrf',
                                choices=['rrf', 'combsum', 'weighted'],
                                help='Fusion method for bidirectional ensemble')
    setwise_parser.add_argument('--alpha', type=float, default=0.5,
                                help='Weight for top-down in weighted fusion (bidirectional only)')
    setwise_parser.add_argument('--gate_strategy', type=str, default='hybrid',
                                choices=['off', 'shortlist', 'uncertain', 'hybrid'],
                                help='When to invoke extra DualEnd logic for selective/bias-aware variants '
                                     '(shortlist routing is ignored for selective heapsort)')
    setwise_parser.add_argument('--shortlist_size', type=int, default=20,
                                help='Prefix depth treated as near the top-k boundary for selective/bias-aware variants')
    setwise_parser.add_argument('--margin_threshold', type=float, default=0.15,
                                help='Backward-compatible alias for the uncertainty percentile; '
                                     '0.15 means the tightest 15%% of query-local BM25-spread windows')
    setwise_parser.add_argument('--uncertainty_percentile', type=float, default=None,
                                help='Query-local percentile cutoff for uncertainty gating; '
                                     '0.15 means the tightest 15%% of BM25-spread windows')
    setwise_parser.add_argument('--order_robust_orderings', type=int, default=3,
                                help='Number of controlled orderings for bias-aware DualEnd windows '
                                     '(bubblesort/selection only)')

    listwise_parser = commands.add_parser('listwise')
    listwise_parser.add_argument('--window_size', type=int, default=3)
    listwise_parser.add_argument('--step_size', type=int, default=1)
    listwise_parser.add_argument('--num_repeat', type=int, default=1)

    args = parse_args(parser, commands)

    if args.run.ir_dataset_name is not None and args.run.pyserini_index is not None:
        raise ValueError('Must specify either --ir_dataset_name or --pyserini_index, not both.')

    arg_dict = vars(args)
    if arg_dict['run'] is None or sum(arg_dict[arg] is not None for arg in arg_dict) != 2:
        raise ValueError('Need to set --run and can only set one of --pointwise, --pairwise, --setwise, --listwise')
    if args.setwise and args.setwise.character_scheme != 'letters_a_w':
        if args.setwise.direction != 'topdown':
            raise ValueError(
                f"--character_scheme {args.setwise.character_scheme} requires --direction topdown; "
                f"got --direction {args.setwise.direction}."
            )
        if args.run.openai_key is not None:
            raise ValueError(
                f"--character_scheme {args.setwise.character_scheme} is not supported with --openai_key."
            )
    main(args)
