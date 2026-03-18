"""
Extended Setwise Ranking Strategies:
1. BottomUpSetwiseLlmRanker - Selects the LEAST relevant document (reverse/bottom-up ranking)
2. DualEndSetwiseLlmRanker - Selects BOTH most and least relevant simultaneously
3. BidirectionalEnsembleRanker - Runs both top-down and bottom-up, fuses rankings

Reference: Extends the setwise approach from Zhuang et al. (SIGIR 2024)
"""

from typing import List, Optional, Tuple
from .rankers import LlmRanker, SearchResult
from .setwise import SetwiseLlmRanker, QWEN_MODEL_TYPES
import copy
import re
import torch
import random
from collections import Counter

random.seed(929)


class BottomUpSetwiseLlmRanker(SetwiseLlmRanker):
    """
    Bottom-Up Setwise Ranker: selects the LEAST relevant document from each comparison set.
    Builds the ranking from the bottom up by iteratively removing the worst documents.

    For top-k ranking (k << n), this requires n-k extractions from a min-heap plus
    k extractions to sort the survivors, totaling n comparisons. This is less efficient
    than standard top-down (k extractions from max-heap) when k << n.
    However, it may exhibit different effectiveness and position bias characteristics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare_worst(self, query: str, docs: List):
        """Select the LEAST relevant document from the candidate set."""
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the least relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the least relevant passage:'

        if self.scoring == 'generation':
            if self.config.model_type == 't5':
                if self.num_permutation == 1:
                    inputs = self._tokenize_inputs(input_text)
                    self.total_prompt_tokens += inputs.input_ids.shape[1]

                    output_ids = self._generate(
                        inputs,
                        max_new_tokens=2,
                        decoder_input_ids=self.decoder_input_ids,
                    )[0]

                    self.total_completion_tokens += output_ids.shape[0]

                    raw_output = self.tokenizer.decode(output_ids,
                                                       skip_special_tokens=True).strip()
                    output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
                    if output is None:
                        output = self._clean_generation_output(raw_output).upper()
                else:
                    id_passage = [(i, p) for i, p in enumerate(docs)]
                    labels = [self.CHARACTERS[i] for i in range(len(docs))]
                    batch_data = []
                    for _ in range(self.num_permutation):
                        batch_data.append([random.sample(id_passage, len(id_passage)),
                                           random.sample(labels, len(labels))])

                    batch_ref = []
                    input_text = []
                    for batch in batch_data:
                        ref = []
                        passages = []
                        characters = []
                        for p, c in zip(batch[0], batch[1]):
                            ref.append(p[0])
                            passages.append(p[1].text)
                            characters.append(c)
                        batch_ref.append((ref, characters))
                        passages = "\n\n".join([f'Passage {characters[i]}: "{passages[i]}"' for i in range(len(passages))])
                        input_text.append(f'Given a query "{query}", which of the following passages is the least relevant one to the query?\n\n' \
                                          + passages + '\n\nOutput only the passage label of the least relevant passage:')

                    inputs = self._tokenize_inputs(input_text, padding=True)
                    self.total_prompt_tokens += inputs.input_ids.shape[1] * inputs.input_ids.shape[0]

                    output_ids = self._generate(
                        inputs,
                        max_new_tokens=2,
                        decoder_input_ids=self.decoder_input_ids.repeat(inputs.input_ids.shape[0], 1),
                    )
                    output = self.tokenizer.batch_decode(output_ids[:, self.decoder_input_ids.shape[1]:],
                                                         skip_special_tokens=True)

                    # vote
                    candidates = []
                    for ref, result in zip(batch_ref, output):
                        docids, characters = ref
                        result = self._parse_single_label(result, characters)
                        if result is None or result not in characters:
                            print(f"Unexpected output: {self._clean_generation_output(str(result))}")
                            continue
                        worst_doc = docids[characters.index(result)]
                        candidates.append(worst_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        candidate_counts = Counter(candidates)
                        max_count = max(candidate_counts.values())
                        most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                                  count == max_count]
                        if len(most_common_candidates) == 1:
                            output = self.CHARACTERS[most_common_candidates[0]]
                        else:
                            output = self.CHARACTERS[random.choice(most_common_candidates)]

            elif self._is_supported_causal_model():
                conversation = [{"role": "user", "content": input_text}]
                prompt = self._build_chat_prompt(conversation)
                prompt += " Passage:"

                inputs = self._tokenize_inputs(prompt)
                self.total_prompt_tokens += inputs.input_ids.shape[1]

                max_new = 256 if self.config.model_type in QWEN_MODEL_TYPES else 4
                output_ids = self._generate(inputs, max_new_tokens=max_new)[0]

                self.total_completion_tokens += output_ids.shape[0]

                raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:],
                                                   skip_special_tokens=False).strip()
                output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
                if output is None:
                    output = self._clean_generation_output(raw_output).upper()

        elif self.scoring == 'likelihood':
            if self.config.model_type == 't5':
                inputs = self._tokenize_inputs(input_text)
                self.total_prompt_tokens += inputs.input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        decoder_input_ids=self.decoder_input_ids,
                    ).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[:len(docs)]]
                    # For bottom-up with "least relevant" prompt: select highest likelihood
                    ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                    output = ranked[0][0]
            else:
                raise NotImplementedError

        if len(output) == 1 and output in self.CHARACTERS:
            pass
        else:
            print(f"Unexpected output: {output}")

        return output

    def heapify_min(self, arr, n, i, query):
        """Min-heapify: select the WORST (least relevant) as root."""
        if self.num_child * i + 1 < n:
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare_worst(query, docs)
            try:
                worst_ind = self.CHARACTERS.index(output)
            except ValueError:
                worst_ind = 0
            try:
                smallest = inds[worst_ind]
            except IndexError:
                smallest = i
            if smallest != i:
                arr[i], arr[smallest] = arr[smallest], arr[i]
                self.heapify_min(arr, n, smallest, query)

    def heapSort(self, arr, query, k):
        """Bottom-up heapsort: build min-heap, extract n-k minimums.
        The remaining k documents are the top-k, then sorted with standard max-heapsort.

        Total work: n-k min-extractions + k max-extractions = n extractions.
        For k=10, n=100: 90 + 10 = 100 extractions (vs. 10 for standard top-down).
        """
        n = len(arr)
        if k >= n:
            # If k >= n, just do full standard heapsort
            super().heapSort(arr, query, k)
            return

        n_to_remove = n - k

        # Build min-heap (worst at root)
        for i in range(n // self.num_child, -1, -1):
            self.heapify_min(arr, n, i, query)

        # Extract n-k minimums (worst documents go to end)
        effective_n = n
        for _ in range(n_to_remove):
            if effective_n <= 1:
                break
            # Swap root (worst) with last unsorted element
            arr[effective_n - 1], arr[0] = arr[0], arr[effective_n - 1]
            effective_n -= 1
            # Re-heapify remaining elements
            self.heapify_min(arr, effective_n, 0, query)

        # Now arr[0..k-1] contains the top-k in min-heap order (unsorted)
        # Sort them using standard max-heapsort (best selection)
        top_k = arr[:k]
        self._sort_top_k(top_k, query)
        arr[:k] = top_k

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            # After _sort_top_k, the top-k are in standard heapsort order (best at end)
            # Reverse to get best-first
            ranking[:self.k] = list(reversed(ranking[:self.k]))

        elif self.method == "bubblesort":
            # Bottom-up bubblesort: sink worst documents to the bottom
            n = len(ranking)
            n_to_remove = n - self.k

            for i in range(n_to_remove):
                # Sink worst to position n-1-i
                target_pos = n - 1 - i
                start_ind = 0
                end_ind = min(self.num_child + 1, target_pos + 1)

                while end_ind <= target_pos + 1 and start_ind < target_pos:
                    window = ranking[start_ind:end_ind]
                    if len(window) < 2:
                        break
                    output = self.compare_worst(query, window)
                    try:
                        worst_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        worst_ind = len(window) - 1
                    worst_ind = min(worst_ind, len(window) - 1)

                    if worst_ind != len(window) - 1:
                        # Move worst to end of window (sink down)
                        actual_worst_pos = start_ind + worst_ind
                        actual_end_pos = start_ind + len(window) - 1
                        ranking[actual_worst_pos], ranking[actual_end_pos] = ranking[actual_end_pos], ranking[actual_worst_pos]

                    start_ind += self.num_child
                    end_ind = min(start_ind + self.num_child + 1, target_pos + 1)
        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results

    def _sort_top_k(self, arr, query):
        """Sort the top-k documents using standard best-selection max-heapsort.
        After this, arr is in ascending order (best at end), caller must reverse.
        """
        n = len(arr)
        if n <= 1:
            return
        # Build max-heap
        for i in range(n // self.num_child, -1, -1):
            self._heapify_standard(arr, n, i, query)
        # Extract all elements
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self._heapify_standard(arr, i, 0, query)

    def _heapify_standard(self, arr, n, i, query):
        """Standard max-heapify using parent class compare (best selection)."""
        if self.num_child * i + 1 < n:
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)  # Uses parent's compare (best selection)
            try:
                best_ind = self.CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self._heapify_standard(arr, n, largest, query)


class DualEndSetwiseLlmRanker(SetwiseLlmRanker):
    """
    Dual-End Setwise Ranker: selects BOTH the most and least relevant documents
    from each comparison set in a single LLM call.

    This extracts 2x information per LLM call, potentially reducing the number
    of required comparisons for the same ranking quality.

    Supports three sorting methods:
    - 'bubblesort': Cocktail shaker sort (bidirectional bubble sort)
    - 'heapsort': Standard heapsort (uses parent compare for heap, bonus worst tracking)
    - 'selection': Double-ended selection sort
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare_both(self, query: str, docs: List) -> Tuple[str, str]:
        """Select both the MOST and LEAST relevant documents from the candidate set.

        Returns:
            Tuple[str, str]: (best_label, worst_label) as character labels
        """
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        if len(docs) < 2:
            return self.CHARACTERS[0], self.CHARACTERS[0]

        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = (
            f'Given a query "{query}", which of the following passages is the most relevant '
            f'and which is the least relevant to the query?\n\n'
            + passages +
            '\n\nOutput only in the format: Best: [label], Worst: [label]'
        )

        best = None
        worst = None

        if self.scoring == 'generation':
            if self.config.model_type == 't5':
                # Use the dual decoder prompt "<pad> Best:" to prime T5 for dual output.
                # With enough max_new_tokens, T5 generates "A, Worst: C" after the prompt.
                inputs = self._tokenize_inputs(input_text)
                self.total_prompt_tokens += inputs.input_ids.shape[1]

                output_ids = self._generate(
                    inputs,
                    max_new_tokens=20,
                    decoder_input_ids=self.dual_decoder_input_ids,
                )[0]

                self.total_completion_tokens += output_ids.shape[0]

                raw_output = self.tokenizer.decode(output_ids,
                                                   skip_special_tokens=True).strip()
                # raw_output includes the decoder prefix, e.g. "Best: A, Worst: C"
                best, worst = self._parse_dual_output(raw_output, len(docs))

            elif self._is_supported_causal_model():
                conversation = [{"role": "user", "content": input_text}]
                prompt = self._build_chat_prompt(conversation)

                inputs = self._tokenize_inputs(prompt)
                self.total_prompt_tokens += inputs.input_ids.shape[1]

                # Thinking models (Qwen3) need a larger budget: the <think>...</think>
                # block can easily consume 200+ tokens before the answer appears.
                # 512 tokens gives ample room for thinking + "Best: A, Worst: C".
                max_new = 512 if self.config.model_type in QWEN_MODEL_TYPES else 64
                output_ids = self._generate(inputs, max_new_tokens=max_new)[0]

                self.total_completion_tokens += output_ids.shape[0]

                output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:],
                                               skip_special_tokens=False).strip()
                output = self._clean_generation_output(output)
                # Always parse from the single call — never fall back to 2 separate calls
                best, worst = self._parse_dual_output(output, len(docs))

        elif self.scoring == 'likelihood':
            if self.config.model_type == 't5':
                # For likelihood mode: use a single forward pass to get all label logits
                # Select highest as best, lowest as worst
                # Note: we use the "most relevant" prompt for likelihood since we extract
                # both best (highest score) and worst (lowest score) from the same distribution
                likelihood_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                                  + passages + '\n\nOutput only the passage label of the most relevant passage:'
                inputs = self._tokenize_inputs(likelihood_text)
                self.total_prompt_tokens += inputs.input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        decoder_input_ids=self.decoder_input_ids,
                    ).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[:len(docs)]]
                    ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                    best = ranked[0][0]
                    worst = ranked[-1][0]
            else:
                raise NotImplementedError(
                    "Likelihood scoring for dual-end is only supported for T5 models. "
                    "Use scoring='generation' for other model types."
                )

        # Safety check: ensure best and worst are different
        if best == worst:
            print(f"Warning: best and worst are the same ({best}), defaulting worst to last character")
            for c in reversed(self.CHARACTERS[:len(docs)]):
                if c != best:
                    worst = c
                    break

        return best, worst

    def _compare_worst_single(self, query: str, docs: List) -> str:
        """Fallback path for models that fail to emit both labels reliably."""
        self.total_compare += 1
        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = (
            f'Given a query "{query}", which of the following passages is the least relevant one to the query?\n\n'
            + passages + '\n\nOutput only the passage label of the least relevant passage:'
        )

        if self.config.model_type == 't5':
            inputs = self._tokenize_inputs(input_text)
            self.total_prompt_tokens += inputs.input_ids.shape[1]
            output_ids = self._generate(
                inputs,
                max_new_tokens=2,
                decoder_input_ids=self.decoder_input_ids,
            )[0]
            self.total_completion_tokens += output_ids.shape[0]
            raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
            if output is None:
                output = self._clean_generation_output(raw_output).upper()
        elif self._is_supported_causal_model():
            conversation = [{"role": "user", "content": input_text}]
            prompt = self._build_chat_prompt(conversation)
            prompt += " Passage:"
            inputs = self._tokenize_inputs(prompt)
            self.total_prompt_tokens += inputs.input_ids.shape[1]
            max_new = 256 if self.config.model_type in QWEN_MODEL_TYPES else 4
            output_ids = self._generate(inputs, max_new_tokens=max_new)[0]
            self.total_completion_tokens += output_ids.shape[0]
            raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()
            output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
            if output is None:
                output = self._clean_generation_output(raw_output).upper()
        else:
            raise NotImplementedError

        if output:
            for char in reversed(output):
                if char in self.CHARACTERS[:len(docs)]:
                    return char
        return self.CHARACTERS[len(docs) - 1]

    def _num_to_label(self, num: int, n_docs: int) -> Optional[str]:
        """Convert a 1-based number to the corresponding passage label."""
        idx = num - 1  # 1-based → 0-based
        if 0 <= idx < n_docs:
            return self.CHARACTERS[idx]
        return None

    def _try_parse_dual_output(self, output: str, n_docs: int) -> Tuple[Optional[str], Optional[str]]:
        """Parse the dual output from the LLM.

        Handles various output formats:
        - "Best: A, Worst: C"      (letter labels)
        - "Best: [A], Worst: [C]"  (bracketed labels)
        - "Best: 1, Worst: 3"      (numeric labels)
        - "A, C"                   (bare letters)
        - "1, 3"                   (bare numbers)
        - "Passage A, Passage C"
        """
        output_upper = self._clean_generation_output(output).upper().strip()
        valid_chars = set(self.CHARACTERS[:n_docs])

        # Label pattern: bare letter or letter in square brackets, e.g. A, [A]
        _L = r'\[?([A-W])\]?'

        # --- Pattern 1: "Best: X, Worst: Y" with letter labels ---
        best_match = re.search(r'BEST[:\s]*(?:PASSAGE\s*)?' + _L, output_upper)
        worst_match = re.search(r'WORST[:\s]*(?:PASSAGE\s*)?' + _L, output_upper)

        if best_match and worst_match:
            best = best_match.group(1)
            worst = worst_match.group(1)
            if best in valid_chars and worst in valid_chars:
                return best, worst

        # --- Pattern 2: "Best: N, Worst: M" with numeric labels (1-based) ---
        best_num_match = re.search(r'BEST[:\s]*(?:PASSAGE\s*)?(\d+)', output_upper)
        worst_num_match = re.search(r'WORST[:\s]*(?:PASSAGE\s*)?(\d+)', output_upper)
        if best_num_match and worst_num_match:
            best = self._num_to_label(int(best_num_match.group(1)), n_docs)
            worst = self._num_to_label(int(worst_num_match.group(1)), n_docs)
            if best is not None and worst is not None:
                return best, worst

        # --- Pattern 3: "most relevant: X ... least relevant: Y" ---
        most_match = re.search(r'MOST\s+RELEVANT[:\s]*(?:PASSAGE\s*)?' + _L, output_upper)
        least_match = re.search(r'LEAST\s+RELEVANT[:\s]*(?:PASSAGE\s*)?' + _L, output_upper)
        if most_match and least_match:
            best = most_match.group(1)
            worst = least_match.group(1)
            if best in valid_chars and worst in valid_chars:
                return best, worst

        # --- Pattern 4: "Passage X" patterns (at least 2) ---
        passage_matches = re.findall(r'PASSAGE\s*' + _L, output_upper)
        passage_matches = [c for c in passage_matches if c in valid_chars]
        if len(passage_matches) >= 2:
            return passage_matches[0], passage_matches[1]

        # --- Pattern 5: comma/space-separated letters or bracketed letters ---
        parts = re.split(r'[,\s]+', output_upper)
        found_chars = []
        for p in parts:
            m = re.fullmatch(r'\[?([A-W])\]?', p)
            if m and m.group(1) in valid_chars:
                found_chars.append(m.group(1))
        if len(found_chars) >= 2:
            return found_chars[0], found_chars[1]

        # --- Pattern 6: two distinct numbers (1-based) anywhere in the output ---
        all_nums = re.findall(r'\b(\d+)\b', output_upper)
        found_num_labels = []
        for n in all_nums:
            label = self._num_to_label(int(n), n_docs)
            if label is not None and (not found_num_labels or label != found_num_labels[-1]):
                found_num_labels.append(label)
            if len(found_num_labels) >= 2:
                return found_num_labels[0], found_num_labels[1]

        return None, None

    def _parse_dual_output(self, output: str, n_docs: int) -> Tuple[str, str]:
        """Parse dual output with guaranteed return — never returns None.

        Falls back to heuristics if _try_parse_dual_output fails:
        1. If one letter found: use it as best, default worst
        2. If one number found: map to label, default the other
        3. Last resort: default to first and last
        """
        best, worst = self._try_parse_dual_output(output, n_docs)
        if best is not None and worst is not None:
            return best, worst

        cleaned = self._clean_generation_output(output)
        output_upper = cleaned.upper().strip()
        valid_chars = set(self.CHARACTERS[:n_docs])

        # Try finding any valid letter characters
        all_found = [c for c in output_upper if c in valid_chars]
        if len(all_found) >= 2 and all_found[0] != all_found[1]:
            print(f"Warning: Partial dual parse from '{cleaned}', using {all_found[0]} as best, {all_found[1]} as worst")
            return all_found[0], all_found[1]
        if len(all_found) >= 1:
            best = all_found[0]
            worst = self.CHARACTERS[n_docs - 1] if best != self.CHARACTERS[n_docs - 1] else self.CHARACTERS[0]
            print(f"Warning: Could only parse one label from '{cleaned}', using {best} as best, {worst} as worst")
            return best, worst

        # Try numeric: find any number that maps to a valid label
        num_match = re.search(r'\b(\d+)\b', cleaned)
        if num_match:
            label = self._num_to_label(int(num_match.group(1)), n_docs)
            if label is not None:
                worst = self.CHARACTERS[n_docs - 1] if label != self.CHARACTERS[n_docs - 1] else self.CHARACTERS[0]
                print(f"Warning: Numeric-only dual parse from '{cleaned}', using {label} as best, {worst} as worst")
                return label, worst

        # Last resort: default to first and last
        print(f"Warning: Could not parse dual output: '{cleaned}', defaulting to A and {self.CHARACTERS[n_docs-1]}")
        return self.CHARACTERS[0], self.CHARACTERS[n_docs - 1]

    def _majority_vote(self, candidates: List[int]) -> int:
        """Return the most common candidate index via majority voting."""
        candidate_counts = Counter(candidates)
        max_count = max(candidate_counts.values())
        most_common = [c for c, count in candidate_counts.items() if count == max_count]
        return most_common[0] if len(most_common) == 1 else random.choice(most_common)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

        if self.method == "heapsort":
            # Use standard heapsort from parent (dual-end doesn't map cleanly to heap)
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))

        elif self.method == "bubblesort":
            # Cocktail shaker sort: properly bidirectional
            self._cocktail_shaker_sort(ranking, query, self.k)

        elif self.method == "selection":
            # Double-ended selection sort
            self._double_ended_selection(ranking, query, self.k)

        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results

    def _cocktail_shaker_sort(self, ranking, query, k):
        """Cocktail shaker sort (bidirectional bubblesort).

        Alternates between:
        1. Backward pass (bottom→top): Compare windows, move best to front and worst to back
        2. Forward pass (top→bottom): Compare windows, move worst to back and best to front

        Each comparison extracts both best and worst from the window, getting 2x
        information per LLM call compared to standard bubblesort.
        """
        n = len(ranking)
        top_sorted = 0  # Number of positions sorted at the top
        bottom_boundary = n  # Upper bound of unsorted region

        while top_sorted < k:
            # === Backward pass (bottom → top): bubble best to position top_sorted ===
            start_ind = bottom_boundary - (self.num_child + 1)
            if start_ind < top_sorted:
                start_ind = top_sorted
            end_ind = start_ind + self.num_child + 1
            if end_ind > bottom_boundary:
                end_ind = bottom_boundary

            worst_sunk = False
            while True:
                if start_ind < top_sorted:
                    start_ind = top_sorted
                window = ranking[start_ind:end_ind]
                if len(window) < 2:
                    break

                best_label, worst_label = self.compare_both(query, window)

                try:
                    best_ind = self.CHARACTERS.index(best_label)
                except ValueError:
                    best_ind = 0
                try:
                    worst_ind = self.CHARACTERS.index(worst_label)
                except ValueError:
                    worst_ind = len(window) - 1

                best_ind = min(best_ind, len(window) - 1)
                worst_ind = min(worst_ind, len(window) - 1)

                # Handle the case where best and worst swap with each other
                if best_ind != 0 or worst_ind != len(window) - 1:
                    # Strategy: first note both positions, then perform swaps carefully
                    actual_best = start_ind + best_ind
                    actual_worst = start_ind + worst_ind
                    front = start_ind
                    back = start_ind + len(window) - 1

                    if actual_best == back and actual_worst == front:
                        # Simple swap: best is at back, worst is at front
                        ranking[front], ranking[back] = ranking[back], ranking[front]
                    else:
                        # Move best to front first
                        if actual_best != front:
                            ranking[front], ranking[actual_best] = ranking[actual_best], ranking[front]
                            # Update worst index if it was at front (now moved to actual_best)
                            if actual_worst == front:
                                actual_worst = actual_best

                        # Move worst to back
                        if actual_worst != back and actual_worst != front:
                            ranking[back], ranking[actual_worst] = ranking[actual_worst], ranking[back]
                            if not worst_sunk:
                                worst_sunk = True

                if start_ind == top_sorted:
                    break

                start_ind -= self.num_child
                end_ind -= self.num_child
                if end_ind < start_ind + 2:
                    break

            top_sorted += 1
            if worst_sunk and bottom_boundary > top_sorted + 1:
                bottom_boundary -= 1

            if top_sorted >= k:
                break

            # === Forward pass (top → bottom): sink worst to bottom_boundary-1 ===
            start_ind = top_sorted
            end_ind = min(start_ind + self.num_child + 1, bottom_boundary)

            best_bubbled = False
            while True:
                window = ranking[start_ind:end_ind]
                if len(window) < 2:
                    break

                best_label, worst_label = self.compare_both(query, window)

                try:
                    best_ind = self.CHARACTERS.index(best_label)
                except ValueError:
                    best_ind = 0
                try:
                    worst_ind = self.CHARACTERS.index(worst_label)
                except ValueError:
                    worst_ind = len(window) - 1

                best_ind = min(best_ind, len(window) - 1)
                worst_ind = min(worst_ind, len(window) - 1)

                actual_best = start_ind + best_ind
                actual_worst = start_ind + worst_ind
                front = start_ind
                back = start_ind + len(window) - 1

                if actual_best == back and actual_worst == front:
                    ranking[front], ranking[back] = ranking[back], ranking[front]
                else:
                    # Move worst to back first (primary goal of forward pass)
                    if actual_worst != back:
                        ranking[back], ranking[actual_worst] = ranking[actual_worst], ranking[back]
                        if actual_best == back:
                            actual_best = actual_worst

                    # Move best to front (bonus)
                    if actual_best != front and actual_best != back:
                        ranking[front], ranking[actual_best] = ranking[actual_best], ranking[front]
                        if not best_bubbled:
                            best_bubbled = True

                start_ind += self.num_child
                end_ind = min(start_ind + self.num_child + 1, bottom_boundary)
                if start_ind >= bottom_boundary:
                    break

            if bottom_boundary > top_sorted + 1:
                bottom_boundary -= 1
            # If best also bubbled up during forward pass, we get a bonus top position
            # (but don't double-count — the backward pass in next iteration will verify)

    def _double_ended_selection(self, ranking, query, k):
        """Double-ended selection sort: extract best and worst per round.

        In each round:
        1. Run a tournament among groups of num_child+1 documents
        2. Compare group winners to find overall best
        3. Compare group losers to find overall worst
        4. Place best at next top position, worst at next bottom position
        5. Continue until k top positions are filled
        """
        n = len(ranking)
        top_idx = 0  # Next top position to fill
        bottom_idx = n - 1  # Next bottom position to fill

        while top_idx < k and top_idx < bottom_idx:
            unsorted_len = bottom_idx - top_idx + 1

            if unsorted_len <= 1:
                break

            if unsorted_len <= self.num_child + 1:
                # Can compare all remaining in one call
                window = ranking[top_idx:bottom_idx + 1]
                best_label, worst_label = self.compare_both(query, window)
                try:
                    best_pos = min(self.CHARACTERS.index(best_label), len(window) - 1)
                except ValueError:
                    best_pos = 0
                try:
                    worst_pos = min(self.CHARACTERS.index(worst_label), len(window) - 1)
                except ValueError:
                    worst_pos = len(window) - 1

                # Convert to absolute indices
                abs_best = top_idx + best_pos
                abs_worst = top_idx + worst_pos

                # Perform swaps carefully to avoid interference
                if abs_best == bottom_idx and abs_worst == top_idx:
                    # Simple swap
                    ranking[top_idx], ranking[bottom_idx] = ranking[bottom_idx], ranking[top_idx]
                else:
                    # Move best to top_idx
                    if abs_best != top_idx:
                        ranking[top_idx], ranking[abs_best] = ranking[abs_best], ranking[top_idx]
                        # Track worst: if worst was at top_idx, it's now at abs_best
                        if abs_worst == top_idx:
                            abs_worst = abs_best

                    # Move worst to bottom_idx
                    if abs_worst != bottom_idx and abs_worst > top_idx:
                        ranking[bottom_idx], ranking[abs_worst] = ranking[abs_worst], ranking[bottom_idx]

                top_idx += 1
                bottom_idx -= 1
            else:
                # Tournament: compare in groups, find overall best and worst
                group_bests = []  # (absolute_index, doc)
                group_worsts = []  # (absolute_index, doc)

                for g_start in range(top_idx, bottom_idx + 1, self.num_child + 1):
                    g_end = min(g_start + self.num_child + 1, bottom_idx + 1)
                    group = ranking[g_start:g_end]
                    if len(group) < 2:
                        group_bests.append((g_start, group[0]))
                        continue

                    best_label, worst_label = self.compare_both(query, group)
                    try:
                        b_idx = min(self.CHARACTERS.index(best_label), len(group) - 1)
                    except ValueError:
                        b_idx = 0
                    try:
                        w_idx = min(self.CHARACTERS.index(worst_label), len(group) - 1)
                    except ValueError:
                        w_idx = len(group) - 1

                    group_bests.append((g_start + b_idx, ranking[g_start + b_idx]))
                    group_worsts.append((g_start + w_idx, ranking[g_start + w_idx]))

                # Find overall best from group winners
                if len(group_bests) > 1:
                    winner_docs_list = [item[1] for item in group_bests]
                    # Compare group winners to find the overall best
                    # May need multiple rounds if more than num_child+1 winners
                    overall_best_tournament_idx = self._tournament_select_best(query, winner_docs_list)
                    overall_best_abs = group_bests[overall_best_tournament_idx][0]
                else:
                    overall_best_abs = group_bests[0][0]

                # Find overall worst from group losers
                if len(group_worsts) > 1:
                    loser_docs_list = [item[1] for item in group_worsts]
                    overall_worst_tournament_idx = self._tournament_select_worst(query, loser_docs_list)
                    overall_worst_abs = group_worsts[overall_worst_tournament_idx][0]
                elif len(group_worsts) == 1:
                    overall_worst_abs = group_worsts[0][0]
                else:
                    overall_worst_abs = bottom_idx

                # Place best at top_idx and worst at bottom_idx
                if overall_best_abs == bottom_idx and overall_worst_abs == top_idx:
                    ranking[top_idx], ranking[bottom_idx] = ranking[bottom_idx], ranking[top_idx]
                else:
                    if overall_best_abs != top_idx:
                        ranking[top_idx], ranking[overall_best_abs] = ranking[overall_best_abs], ranking[top_idx]
                        if overall_worst_abs == top_idx:
                            overall_worst_abs = overall_best_abs

                    if overall_worst_abs != bottom_idx and overall_worst_abs > top_idx:
                        ranking[bottom_idx], ranking[overall_worst_abs] = ranking[overall_worst_abs], ranking[bottom_idx]

                top_idx += 1
                bottom_idx -= 1

    def _tournament_select_best(self, query, docs_list):
        """Run a tournament to find the best document from a list. Returns index in docs_list."""
        if len(docs_list) <= self.num_child + 1:
            best_label = self.compare(query, docs_list)
            try:
                return min(self.CHARACTERS.index(best_label), len(docs_list) - 1)
            except ValueError:
                return 0

        # Multiple rounds needed
        current_indices = list(range(len(docs_list)))
        while len(current_indices) > 1:
            next_round = []
            for g in range(0, len(current_indices), self.num_child + 1):
                group_indices = current_indices[g:g + self.num_child + 1]
                if len(group_indices) == 1:
                    next_round.append(group_indices[0])
                    continue
                group_docs = [docs_list[idx] for idx in group_indices]
                best_label = self.compare(query, group_docs)
                try:
                    winner = min(self.CHARACTERS.index(best_label), len(group_docs) - 1)
                except ValueError:
                    winner = 0
                next_round.append(group_indices[winner])
            current_indices = next_round
        return current_indices[0]

    def _tournament_select_worst(self, query, docs_list):
        """Run a tournament to find the worst document from a list. Returns index in docs_list."""
        if len(docs_list) <= self.num_child + 1:
            # Use worst-selection prompt
            self.total_compare += 1
            passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs_list)])
            worst_text = f'Given a query "{query}", which of the following passages is the least relevant one to the query?\n\n' \
                         + passages + '\n\nOutput only the passage label of the least relevant passage:'

            if self.config.model_type == 't5':
                inputs = self._tokenize_inputs(worst_text)
                self.total_prompt_tokens += inputs.input_ids.shape[1]
                output_ids = self._generate(
                    inputs,
                    max_new_tokens=2,
                    decoder_input_ids=self.decoder_input_ids,
                )[0]
                self.total_completion_tokens += output_ids.shape[0]
                raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs_list)])
                if output is None:
                    output = self._clean_generation_output(raw_output).upper()
            elif self._is_supported_causal_model():
                conversation = [{"role": "user", "content": worst_text}]
                prompt = self._build_chat_prompt(conversation)
                prompt += " Passage:"
                inputs = self._tokenize_inputs(prompt)
                self.total_prompt_tokens += inputs.input_ids.shape[1]
                max_new = 256 if self.config.model_type in QWEN_MODEL_TYPES else 4
                output_ids = self._generate(inputs, max_new_tokens=max_new)[0]
                self.total_completion_tokens += output_ids.shape[0]
                raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()
                output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs_list)])
                if output is None:
                    output = self._clean_generation_output(raw_output).upper()
            else:
                return 0

            try:
                return min(self.CHARACTERS.index(output.upper()), len(docs_list) - 1)
            except ValueError:
                return len(docs_list) - 1

        # Multiple rounds
        current_indices = list(range(len(docs_list)))
        while len(current_indices) > 1:
            next_round = []
            for g in range(0, len(current_indices), self.num_child + 1):
                group_indices = current_indices[g:g + self.num_child + 1]
                if len(group_indices) == 1:
                    next_round.append(group_indices[0])
                    continue
                group_docs = [docs_list[idx] for idx in group_indices]
                worst_idx = self._tournament_select_worst(query, group_docs)
                next_round.append(group_indices[worst_idx])
            current_indices = next_round
        return current_indices[0]


class BidirectionalEnsembleRanker(LlmRanker):
    """
    Bidirectional Ensemble Ranker: runs both standard (top-down) and bottom-up
    setwise ranking independently, then fuses the rankings.

    Fusion methods:
    - 'rrf': Reciprocal Rank Fusion
    - 'combsum': Normalized score summation
    - 'weighted': Weighted combination (alpha * topdown + (1-alpha) * bottomup)
    """

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method='heapsort',
                 num_permutation=1,
                 fusion='rrf',
                 alpha=0.5,
                 cache_dir=None):

        self.k = k
        self.fusion = fusion
        self.alpha = alpha

        # Create top-down ranker first
        self.topdown_ranker = SetwiseLlmRanker(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            device=device,
            num_child=num_child,
            k=k,
            scoring=scoring,
            method=method,
            num_permutation=num_permutation,
            cache_dir=cache_dir
        )

        # Create bottom-up ranker that shares model weights with top-down
        # We avoid double model loading by creating a minimal instance and
        # copying the model reference
        self.bottomup_ranker = BottomUpSetwiseLlmRanker.__new__(BottomUpSetwiseLlmRanker)
        # Copy all attributes from topdown_ranker
        self.bottomup_ranker.device = self.topdown_ranker.device
        self.bottomup_ranker.num_child = self.topdown_ranker.num_child
        self.bottomup_ranker.num_permutation = self.topdown_ranker.num_permutation
        self.bottomup_ranker.k = self.topdown_ranker.k
        self.bottomup_ranker.config = self.topdown_ranker.config
        self.bottomup_ranker.tokenizer = self.topdown_ranker.tokenizer
        self.bottomup_ranker.llm = self.topdown_ranker.llm
        self.bottomup_ranker.scoring = self.topdown_ranker.scoring
        self.bottomup_ranker.method = self.topdown_ranker.method
        self.bottomup_ranker.total_compare = 0
        self.bottomup_ranker.total_completion_tokens = 0
        self.bottomup_ranker.total_prompt_tokens = 0
        if hasattr(self.topdown_ranker, 'decoder_input_ids'):
            self.bottomup_ranker.decoder_input_ids = self.topdown_ranker.decoder_input_ids
        if hasattr(self.topdown_ranker, 'dual_decoder_input_ids'):
            self.bottomup_ranker.dual_decoder_input_ids = self.topdown_ranker.dual_decoder_input_ids
        if hasattr(self.topdown_ranker, 'target_token_ids'):
            self.bottomup_ranker.target_token_ids = self.topdown_ranker.target_token_ids
        self.bottomup_ranker.max_input_tokens = self.topdown_ranker.max_input_tokens
        self.bottomup_ranker._warned_input_truncation = False

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        ranking_copy1 = copy.deepcopy(ranking)
        ranking_copy2 = copy.deepcopy(ranking)

        # Run top-down
        topdown_results = self.topdown_ranker.rerank(query, ranking_copy1)

        # Run bottom-up
        bottomup_results = self.bottomup_ranker.rerank(query, ranking_copy2)

        # Aggregate stats
        self.total_compare = self.topdown_ranker.total_compare + self.bottomup_ranker.total_compare
        self.total_prompt_tokens = self.topdown_ranker.total_prompt_tokens + self.bottomup_ranker.total_prompt_tokens
        self.total_completion_tokens = self.topdown_ranker.total_completion_tokens + self.bottomup_ranker.total_completion_tokens

        # Fuse rankings
        if self.fusion == 'rrf':
            return self._rrf_fusion(topdown_results, bottomup_results)
        elif self.fusion == 'combsum':
            return self._combsum_fusion(topdown_results, bottomup_results)
        elif self.fusion == 'weighted':
            return self._weighted_fusion(topdown_results, bottomup_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

    def _rrf_fusion(self, results1: List[SearchResult], results2: List[SearchResult],
                    rrf_k: int = 60) -> List[SearchResult]:
        """Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i(d)))"""
        scores = {}

        for rank, doc in enumerate(results1, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + 1.0 / (rrf_k + rank)

        for rank, doc in enumerate(results2, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + 1.0 / (rrf_k + rank)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (docid, score) in enumerate(sorted_docs, 1):
            results.append(SearchResult(docid=docid, score=-rank, text=None))

        return results

    def _combsum_fusion(self, results1: List[SearchResult], results2: List[SearchResult]) -> List[SearchResult]:
        """CombSUM: normalize scores per run, then sum."""
        scores = {}

        n1 = len(results1)
        for rank, doc in enumerate(results1, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + (n1 - rank + 1) / n1

        n2 = len(results2)
        for rank, doc in enumerate(results2, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + (n2 - rank + 1) / n2

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (docid, score) in enumerate(sorted_docs, 1):
            results.append(SearchResult(docid=docid, score=-rank, text=None))

        return results

    def _weighted_fusion(self, results1: List[SearchResult], results2: List[SearchResult]) -> List[SearchResult]:
        """Weighted fusion: alpha * topdown_score + (1-alpha) * bottomup_score."""
        scores = {}

        n1 = len(results1)
        for rank, doc in enumerate(results1, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + self.alpha * (n1 - rank + 1) / n1

        n2 = len(results2)
        for rank, doc in enumerate(results2, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + (1 - self.alpha) * (n2 - rank + 1) / n2

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (docid, score) in enumerate(sorted_docs, 1):
            results.append(SearchResult(docid=docid, score=-rank, text=None))

        return results

    def truncate(self, text, length):
        return self.topdown_ranker.truncate(text, length)
