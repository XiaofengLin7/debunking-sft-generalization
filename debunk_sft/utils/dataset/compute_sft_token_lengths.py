"""
Compute average token lengths for prompt and response fields in SFT parquet datasets.

This mirrors the tokenization behavior of `thirdparty/verl/verl/utils/dataset/sft_dataset.py`:
- Optionally apply the tokenizer's chat template to the prompt with `add_generation_prompt=True`.
- Append the tokenizer's EOS token to the response.
- Tokenize both with `add_special_tokens=False`.

Usage:
  python -m reil.utils.dataset.compute_sft_token_lengths \
    --parquet /path/to/data.parquet \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt-key prompt \
    --response-key response \
    --chat-template

Multiple parquet files are supported by passing `--parquet` multiple times.
If your prompt/response columns contain dicts and you need to pick nested values,
pass `--prompt-dict-keys key1 --prompt-dict-keys key2` (applied sequentially),
and similarly for `--response-dict-keys`.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _series_to_item(value):
    """Mimic SFTDataset.series_to_item: unwrap single-element Series/ndarray."""
    try:
        import pandas  # type: ignore
        import numpy  # type: ignore
    except Exception:
        pandas = None  # type: ignore
        numpy = None  # type: ignore

    while (
        (pandas is not None and isinstance(value, pandas.core.series.Series))
        or (numpy is not None and isinstance(value, numpy.ndarray))
    ) and len(value) == 1:
        value = value[0]
    return value


def _apply_dict_keys(series: pd.Series, keys: Sequence[str]) -> pd.Series:
    if not keys:
        return series
    current = series
    for key in keys:
        current = current.apply(lambda x: _series_to_item(x)[key])
    return current


def _load_prompts_and_responses(
    parquet_files: Sequence[str],
    prompt_key: str,
    prompt_dict_keys: Sequence[str],
    response_key: str,
    response_dict_keys: Sequence[str],
) -> Tuple[List[str], List[str]]:
    dataframes: List[pd.DataFrame] = []
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        dataframes.append(df)
    dataframe = pd.concat(dataframes, ignore_index=True) if len(dataframes) > 1 else dataframes[0]

    prompts_series = dataframe[prompt_key]
    prompts_series = _apply_dict_keys(prompts_series, prompt_dict_keys)
    responses_series = dataframe[response_key]
    responses_series = _apply_dict_keys(responses_series, response_dict_keys)

    prompts: List[str] = prompts_series.tolist()
    responses: List[str] = responses_series.tolist()
    return prompts, responses


def _build_prompt_string(tokenizer: PreTrainedTokenizerBase, prompt_text: str, use_chat_template: bool) -> str:
    if use_chat_template:
        chat = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)  # type: ignore[attr-defined]
        except Exception:
            # Fallback if tokenizer doesn't implement chat templates
            return prompt_text
    return prompt_text


def _build_response_string(tokenizer: PreTrainedTokenizerBase, response_text: str) -> str:
    eos: str = tokenizer.eos_token or ""
    return response_text + eos


def _token_length(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    # HF returns List[int] or List[List[int]] depending on batch; ensure scalar length
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def compute_average_token_lengths(
    parquet_files: Sequence[str],
    tokenizer_name_or_path: str,
    prompt_key: str = "prompt",
    prompt_dict_keys: Sequence[str] = (),
    response_key: str = "response",
    response_dict_keys: Sequence[str] = (),
    use_chat_template: bool = False,
    sample_limit: int | None = None,
) -> Tuple[float, float, int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

    prompts, responses = _load_prompts_and_responses(
        parquet_files=parquet_files,
        prompt_key=prompt_key,
        prompt_dict_keys=list(prompt_dict_keys),
        response_key=response_key,
        response_dict_keys=list(response_dict_keys),
    )

    if sample_limit is not None:
        prompts = prompts[:sample_limit]
        responses = responses[:sample_limit]

    total_prompt_tokens = 0
    total_response_tokens = 0
    num_samples = 0

    for prompt_text, response_text in zip(prompts, responses):
        if prompt_text is None or response_text is None:
            continue
        try:
            prompt_str = _build_prompt_string(tokenizer, str(prompt_text), use_chat_template)
            response_str = _build_response_string(tokenizer, str(response_text))

            prompt_len = _token_length(tokenizer, prompt_str)
            response_len = _token_length(tokenizer, response_str)
        except Exception:
            # Skip malformed rows
            continue

        total_prompt_tokens += prompt_len
        total_response_tokens += response_len
        num_samples += 1

    if num_samples == 0:
        return 0.0, 0.0, 0

    avg_prompt = total_prompt_tokens / float(num_samples)
    avg_response = total_response_tokens / float(num_samples)
    return avg_prompt, avg_response, num_samples


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute average token lengths for SFT parquet datasets")
    parser.add_argument(
        "--parquet",
        dest="parquet_files",
        action="append",
        required=True,
        help="Path to a parquet file. Pass multiple times to include more files.",
    )
    parser.add_argument(
        "--tokenizer",
        dest="tokenizer_name_or_path",
        required=True,
        help="HF tokenizer name or local path",
    )
    parser.add_argument("--prompt-key", default="prompt", help="Dataframe column for prompts")
    parser.add_argument(
        "--prompt-dict-keys",
        dest="prompt_dict_keys",
        action="append",
        default=[],
        help="Nested keys to select prompt value inside dict-structured columns. Can be repeated.",
    )
    parser.add_argument("--response-key", default="response", help="Dataframe column for responses")
    parser.add_argument(
        "--response-dict-keys",
        dest="response_dict_keys",
        action="append",
        default=[],
        help="Nested keys to select response value inside dict-structured columns. Can be repeated.",
    )
    parser.add_argument(
        "--chat-template",
        dest="use_chat_template",
        action="store_true",
        help="Apply tokenizer chat template to prompts (add_generation_prompt=True)",
    )
    parser.add_argument(
        "--limit",
        dest="sample_limit",
        type=int,
        default=None,
        help="Optionally limit number of samples for speed",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    avg_prompt, avg_response, count = compute_average_token_lengths(
        parquet_files=args.parquet_files,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        prompt_key=args.prompt_key,
        prompt_dict_keys=args.prompt_dict_keys,
        response_key=args.response_key,
        response_dict_keys=args.response_dict_keys,
        use_chat_template=args.use_chat_template,
        sample_limit=args.sample_limit,
    )
    print(f"samples\t{count}")
    print(f"avg_prompt_tokens\t{avg_prompt:.4f}")
    print(f"avg_response_tokens\t{avg_response:.4f}")


if __name__ == "__main__":
    main()


