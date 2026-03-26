from typing import List, Optional, Sequence
from .rankers import LlmRanker, SearchResult
import openai
import os
import time
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import copy
from collections import Counter
import tiktoken
import random

_DEBUG = os.environ.get("LLM_RANKER_DEBUG", "").lower() in ("1", "true", "yes")
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_IMPORT_ERROR = None
except ImportError as err:
    LLM = None
    SamplingParams = None
    LoRARequest = None
    VLLM_IMPORT_ERROR = err

random.seed(929)

CAUSAL_MODEL_TYPES = {"llama", "qwen2", "qwen3", "qwen3_moe", "qwen3_5"}
QWEN_MODEL_TYPES = {"qwen2", "qwen3", "qwen3_moe", "qwen3_5"}


class SetwiseLlmRanker(LlmRanker):
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None):

        self.device = device
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir,
                                                   trust_remote_code=True)
        self._warned_input_truncation = False
        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                         if tokenizer_name_or_path is not None else
                                                         model_name_or_path,
                                                         cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.device) if self.tokenizer else None
            self.dual_decoder_input_ids = self.tokenizer.encode("<pad> Best:",
                                                                return_tensors="pt",
                                                                add_special_tokens=False).to(self.device) if self.tokenizer else None

            self.target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {self.CHARACTERS[i]}'
                                                                      for i in range(len(self.CHARACTERS))],
                                                                     return_tensors="pt",
                                                                     add_special_tokens=False,
                                                                     padding=True).input_ids[:, -1]
        elif self._is_supported_causal_model():
            tokenizer_kwargs = {"cache_dir": cache_dir}
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": "auto" if device == "cuda" else torch.float32,
                "cache_dir": cache_dir,
            }
            if self.config.model_type in QWEN_MODEL_TYPES:
                tokenizer_kwargs["trust_remote_code"] = True
                model_kwargs["trust_remote_code"] = True

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
            if hasattr(self.tokenizer, "use_default_system_prompt"):
                self.tokenizer.use_default_system_prompt = False
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if 'vicuna' and 'v1.5' in model_name_or_path:
                self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            ).eval()
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} is not supported yet for setwise:(")

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.max_input_tokens = self._resolve_max_input_tokens()

    def _is_supported_causal_model(self):
        return self.config.model_type in CAUSAL_MODEL_TYPES

    def _chat_template_kwargs(self):
        if self.config.model_type in QWEN_MODEL_TYPES:
            return {"enable_thinking": False}
        return {}

    def _build_chat_prompt(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs(),
        )

    def _resolve_max_input_tokens(self):
        candidates = []
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and 0 < model_max_length < 100000:
            candidates.append(model_max_length)
        for attr in ("n_positions", "max_position_embeddings"):
            value = getattr(self.config, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        return min(candidates) if candidates else None

    def _tokenize_inputs(self, inputs, padding=False):
        raw = self.tokenizer(inputs, add_special_tokens=True, return_attention_mask=True)
        raw_ids = raw["input_ids"]
        if raw_ids and isinstance(raw_ids[0], int):
            lengths = [len(raw_ids)]
        else:
            lengths = [len(ids) for ids in raw_ids]

        if (
            self.max_input_tokens is not None
            and lengths
            and max(lengths) > self.max_input_tokens
            and not self._warned_input_truncation
        ):
            print(
                f"Warning: prompt length {max(lengths)} exceeds model limit {self.max_input_tokens}; "
                "truncating the encoded prompt. Lower --passage_length or --query_length for cleaner comparisons."
            )
            self._warned_input_truncation = True

        kwargs = {
            "return_tensors": "pt",
            "return_attention_mask": True,
            "truncation": self.max_input_tokens is not None,
        }
        if self.max_input_tokens is not None:
            kwargs["max_length"] = self.max_input_tokens
        if padding:
            kwargs["padding"] = True
        return self.tokenizer(inputs, **kwargs).to(self.device)

    def _generate(self, model_inputs, max_new_tokens, decoder_input_ids=None):
        kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
        }
        if decoder_input_ids is not None:
            kwargs["decoder_input_ids"] = decoder_input_ids
        if self._is_supported_causal_model():
            generation_config = copy.deepcopy(getattr(self.llm, "generation_config", None))
            if generation_config is not None:
                generation_config.do_sample = False
                for attr in (
                    "temperature",
                    "top_k",
                    "top_p",
                    "min_p",
                    "typical_p",
                    "epsilon_cutoff",
                    "eta_cutoff",
                ):
                    if hasattr(generation_config, attr):
                        setattr(generation_config, attr, None)
                if self.tokenizer.pad_token_id is not None:
                    generation_config.pad_token_id = self.tokenizer.pad_token_id
                if self.tokenizer.eos_token_id is not None:
                    generation_config.eos_token_id = self.tokenizer.eos_token_id
                kwargs["generation_config"] = generation_config
            else:
                kwargs["do_sample"] = False
                if self.tokenizer.pad_token_id is not None:
                    kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                if self.tokenizer.eos_token_id is not None:
                    kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        return self.llm.generate(**kwargs)

    def _clean_generation_output(self, output: str) -> str:
        stripped = output.strip()

        # --- Qwen-family thinking tokens (<think>...</think>) ---
        # Strip complete blocks first, then any truncated/unclosed block
        cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", stripped)
        cleaned = re.sub(r"(?is)<think>.*", "", cleaned)

        # --- Pipe-delimited special tokens (Qwen / LLaMA / general) ---
        # Matches <|im_start|>, <|im_end|>, <|endoftext|>, <|end|>, etc.
        cleaned = re.sub(r"<\|[^|]*\|>", " ", cleaned)

        # --- T5 / Flan-T5 special tokens ---
        cleaned = re.sub(r"</s>", " ", cleaned)
        cleaned = re.sub(r"<pad>", " ", cleaned)
        cleaned = re.sub(r"<unk>", " ", cleaned)
        cleaned = re.sub(r"<extra_id_\d+>", " ", cleaned)

        # --- General XML-style tags left over from any tokenizer ---
        # Strip isolated angle-bracket tokens like <s>, </s>, <bos>, <eos>, etc.
        cleaned = re.sub(r"</?(?:s|bos|eos|sep|cls|mask|pad)\s*>", " ", cleaned)

        cleaned = cleaned.strip()
        return cleaned or stripped

    def _parse_single_label(self, output: str, valid_chars: Sequence[str]) -> Optional[str]:
        cleaned = self._clean_generation_output(output)
        output_upper = cleaned.upper()
        valid = set(valid_chars)

        for pattern in (
            r"(?:BEST|WORST|MOST\s+RELEVANT|LEAST\s+RELEVANT|ANSWER|OUTPUT)\s*[:\-\s]*(?:PASSAGE\s*)?\[?([A-W])\]?",
            r"PASSAGE\s*\[?([A-W])\]?",
        ):
            for match in re.findall(pattern, output_upper):
                if match in valid:
                    return match

        # Match standalone letter or bracketed letter like [A]
        for match in re.findall(r"(?:\[([A-W])\]|\b([A-W])\b)", output_upper):
            char = match[0] or match[1]
            if char in valid:
                return char

        all_found = [char for char in output_upper if char in valid]
        if all_found and len(set(all_found)) == 1:
            return all_found[0]

        # Handle numeric outputs: map 1-based index to the corresponding label
        num_match = re.search(r"\b(\d+)\b", cleaned)
        if num_match:
            idx = int(num_match.group(1)) - 1  # convert 1-based to 0-based
            if 0 <= idx < len(valid_chars):
                return valid_chars[idx]

        # Handle refusal outputs: model says none are relevant — default to first
        # passage (maintains current order, equivalent to no swap)
        if re.search(
            r"(?i)\b(none of the|no passage|not relevant|none are|cannot determine|neither|none|i cannot)\b",
            cleaned,
        ):
            return valid_chars[0]

        return None

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage:'

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
                        input_text.append(f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                                          + passages + '\n\nOutput only the passage label of the most relevant passage:')

                    inputs = self._tokenize_inputs(input_text, padding=True)
                    self.total_prompt_tokens += inputs.input_ids.shape[1] * inputs.input_ids.shape[0]

                    output_ids = self._generate(
                        inputs,
                        max_new_tokens=2,
                        decoder_input_ids=self.decoder_input_ids.repeat(inputs.input_ids.shape[0], 1),
                    )
                    self.total_completion_tokens += output_ids.shape[0] * (output_ids.shape[1] - self.decoder_input_ids.shape[1])

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
                        win_doc = docids[characters.index(result)]
                        candidates.append(win_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        # handle tie
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

                # Thinking models (e.g. Qwen3) may emit <think>...</think> before the
                # answer, so we need enough token budget for the thinking block to
                # complete.  Non-thinking models will hit EOS well before 256 tokens.
                max_new = 256 if self.config.model_type in QWEN_MODEL_TYPES else 4
                output_ids = self._generate(inputs, max_new_tokens=max_new)[0]

                self.total_completion_tokens += output_ids.shape[0] - inputs.input_ids.shape[1]

                # Decode WITHOUT skipping special tokens so <think>...</think> tags
                # are preserved for proper stripping by _clean_generation_output.
                # With skip_special_tokens=True, <think>/<think> are removed but
                # the thinking *content* leaks through and pollutes parsing.
                raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:],
                                                   skip_special_tokens=False).strip()
                if _DEBUG:
                    cleaned_dbg = self._clean_generation_output(raw_output)
                    print(f"[DEBUG] raw={repr(raw_output[:200])}  cleaned={repr(cleaned_dbg[:200])}")
                output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
                if output is None:
                    output = self._clean_generation_output(raw_output).upper()

        elif self.scoring == 'likelihood':
            if self.config.model_type == 't5':
                inputs = self._tokenize_inputs(input_text)
                self.total_prompt_tokens += inputs.input_ids.shape[1]
                # Completion tokens = 0: likelihood reads logits from a single forward
                # pass — no tokens are generated (no autoregressive decoding).
                with torch.no_grad():
                    logits = self.llm(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        decoder_input_ids=self.decoder_input_ids,
                    ).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[:len(docs)]]
                    ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                    output = ranked[0][0]

            else:
                raise NotImplementedError

        if len(output) == 1 and output in self.CHARACTERS:
            pass
        else:
            print(f"Unexpected output: {output}")

        return output

    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)
            try:
                best_ind = self.CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query)

    def rerank(self,  query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        
        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)

            for i in range(self.k):
                # Clamp last_start so the window always contains at least
                # num_child+1 documents (or all remaining documents) when
                # start_ind is clamped to i.
                if last_start < i:
                    last_start = i
                start_ind = last_start
                end_ind = min(last_start + (self.num_child + 1), len(ranking))
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    # Need at least 2 documents for a meaningful comparison
                    if end_ind - start_ind < 2:
                        break
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
                        if not is_change:
                            is_change = True
                            if last_start != len(ranking) - (self.num_child + 1) \
                                    and best_ind == len(ranking[start_ind:end_ind])-1:
                                last_start += len(ranking[start_ind:end_ind])-1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child
                    
        ##  this is a bit slower but standard bobblesort implementation, keep here FYI
        # elif self.method == "bubblesort":
        #     for i in range(k):
        #         start_ind = len(ranking) - (self.num_child + 1)
        #         end_ind = len(ranking)
        #         while True:
        #             if start_ind < i:
        #                 start_ind = i
        #             output = self.compare(query, ranking[start_ind:end_ind])
        #             try:
        #                 best_ind = self.CHARACTERS.index(output)
        #             except ValueError:
        #                 best_ind = 0
        #             if best_ind != 0:
        #                 ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
        #
        #             if start_ind == i:
        #                 break
        #
        #             start_ind -= self.num_child
        #             end_ind -= self.num_child

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

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class OpenAiSetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(self, model_name_or_path, api_key, num_child=3, method='heapsort', k=10):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.num_child = num_child
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."
        openai.api_key = api_key

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage.'

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0.0,
                    request_timeout=15
                )

                self.total_completion_tokens += int(response['usage']['completion_tokens'])
                self.total_prompt_tokens += int(response['usage']['prompt_tokens'])

                output = response['choices'][0]['message']['content']
                matches = re.findall(r"(Passage [A-Z])", output, re.MULTILINE)
                if matches:
                    output = matches[0][8]
                elif output.strip() in self.CHARACTERS:
                    pass
                else:
                    print(f"Unexpected output: {output}")
                    output = "A"
                return output

            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(5)
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(5)
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(5)
                continue
            except openai.error.InvalidRequestError as e:
                # Handle invalid request error
                print(f"OpenAI API request was invalid: {e}")
                raise e
            except openai.error.AuthenticationError as e:
                # Handle authentication error
                print(f"OpenAI API request failed authentication: {e}")
                raise e
            except openai.error.Timeout as e:
                # Handle timeout error
                print(f"OpenAI API request timed out: {e}")
                time.sleep(5)
                continue
            except openai.error.ServiceUnavailableError as e:
                # Handle service unavailable error
                print(f"OpenAI API request failed with a service unavailable error: {e}")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Unknown error: {e}")
                raise e

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])



class RankR1SetwiseLlmRanker(SetwiseLlmRanker):
    CHARACTERS = [f'[{i+1}]' for i in range(20)]

    def __init__(self,
                 model_name_or_path,
                 prompt_file,
                 lora_name_or_path=None,
                 tokenizer_name_or_path=None,
                 num_child=19,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None,
                 verbose=False):

        if VLLM_IMPORT_ERROR is not None:
            raise ImportError(
                "vllm is required for RankR1SetwiseLlmRanker. Install vllm to use this ranker."
            ) from VLLM_IMPORT_ERROR

        if scoring != 'generation':
            raise NotImplementedError(f"Scoring method {scoring} is not supported for RankR1SetwiseLlmRanker. RankR1SetwiseLlmRanker only supports 'generation' scoring.")
        self.verbose = verbose

        import toml
        self.prompt = toml.load(prompt_file)

        from huggingface_hub import snapshot_download
        import os
        if lora_name_or_path is not None:
            # check if the path exists
            if not os.path.exists(lora_name_or_path):
                # download the model
                lora_path = snapshot_download(lora_name_or_path)
            else:
                lora_path = lora_name_or_path
        else:
            lora_path = None

        self.lora_path = lora_path
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.sampling_params = SamplingParams(temperature=0.0,
                                              max_tokens=2048)
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
        self.llm = LLM(model=model_name_or_path,
                       tokenizer=tokenizer_name_or_path,
                       enable_lora=True if lora_name_or_path is not None else False,
                       max_lora_rank=32,
                       )

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        id_passage = [(i, p) for i, p in enumerate(docs)]
        labels = [self.CHARACTERS[i] for i in range(len(docs))]
        batch_data = []
        for _ in range(self.num_permutation):
            batch_data.append([random.sample(id_passage, len(id_passage)),
                               labels])

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
            passages = "\n".join([f'{characters[i]} {passages[i]}' for i in range(len(passages))])
            system_message = self.prompt["prompt_system"]
            user_message = self.prompt['prompt_user'].format(query=query,
                                                             docs=passages)
            input_text.append([
                {'role': "system", 'content': system_message},
                {'role': "user", 'content': user_message}
            ])
        outputs = self.llm.chat(input_text,
                                sampling_params=self.sampling_params,
                                use_tqdm=False,
                                lora_request=LoRARequest("R1adapter",
                                                         1,
                                                         self.lora_path)
                                if self.lora_path is not None else None,
                                )
        results = []
        for output, input in zip(outputs, input_text):
            self.total_completion_tokens += len(output.outputs[0].token_ids)
            self.total_prompt_tokens += len(output.prompt_token_ids)

            completion = output.outputs[0].text

            if self.verbose:
                print('--------------------------------------')
                print(f'query: {query}')
                print(f'input_text:\n{self.tokenizer.apply_chat_template(input, tokenize=False)}')
                print(f'completion:\n{completion}')
                print('--------------------------------------')

            pattern = rf'{self.prompt["pattern"]}'
            match = re.search(pattern, completion.lower(), re.DOTALL)
            if match:
                results.append(match.group(1).strip())
            else:
                results.append(f'input_text:\n{input}, completion:\n{completion}')

        # vote
        candidates = []
        for ref, result in zip(batch_ref, results):
            result = result.strip()
            docids, characters = ref
            if result not in characters:
                if self.verbose:
                    print(f"Unexpected output: {result}")
                continue
            win_doc = docids[characters.index(result)]
            candidates.append(win_doc)

        if len(candidates) == 0:
            if self.verbose:
                print(f"Unexpected voting: {results}")
            output = "Unexpected voting."
        else:
            # handle tie
            candidate_counts = Counter(candidates)
            max_count = max(candidate_counts.values())
            most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                      count == max_count]
            if len(most_common_candidates) == 1:
                output = self.CHARACTERS[most_common_candidates[0]]
            else:
                output = self.CHARACTERS[random.choice(most_common_candidates)]

        if output in self.CHARACTERS:
            pass
        else:
            if self.verbose:
                print(f"Unexpected output: {output}")

        return output
