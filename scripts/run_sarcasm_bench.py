import datetime
import os
from functools import partial
from pathlib import Path
from typing import Optional, cast

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    enqueue=True,
    colorize=True,
)
curent_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logger.add(
    f"run_sarcasm_bench_{curent_datetime}.log",
    serialize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    enqueue=True,
)

import click

from sarcbench.generator_map_functions import (
    openai_requests_map_func,
    vllm_requests_map_func,
)


@click.group(chain=True, invoke_without_command=False)
@click.option("--dataset-path", type=str, help="Dataset to use")
@click.option("--dataset-name", type=str, required=False, help="Dataset name")
@click.option("--dataset-split", type=str, required=False, help="Dataset split")
@click.option("--config-file-path", type=str, help="Config file path")
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output path",
)
@click.option(
    "--num-debug-samples",
    type=int,
    default=-1,
    help="Number of debug samples (for testing)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for reproducibility",
)
def run_sarcasm_bench(
    dataset_path: str,
    dataset_name: Optional[str],
    dataset_split: Optional[str],
    config_file_path: str,
    output_path: Path,
    num_debug_samples: int,
    seed: int,
) -> None:
    pass


@run_sarcasm_bench.result_callback()
def process_pipeline(
    processors,
    dataset_path: str,
    dataset_name: str,
    dataset_split: str,
    config_file_path: str,
    output_path: Path,
    num_debug_samples: int,
    seed: int,
):
    try:
        datasets = load_dataset(dataset_path, dataset_name, split=dataset_split)
    except ValueError:
        datasets = load_from_disk(dataset_path)

    if isinstance(datasets, DatasetDict):
        datasets = datasets
    else:
        datasets = {"default": datasets}

    finished_datasets = {}
    for key, dataset in datasets.items():
        dataset = cast(Dataset, dataset)
        logger.info(f"evaluating dataset ({dataset_name}, {key})")

        if num_debug_samples > 0:
            dataset = dataset.select(range(num_debug_samples))

        for processor in processors:
            dataset = processor(dataset, config_file_path, seed)

        finished_datasets[key] = dataset

    output_path.mkdir(parents=True, exist_ok=True)
    for key, dataset in finished_datasets.items():
        save_path = (
            output_path
            / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{key}"
        )
        dataset.save_to_disk(save_path)
        logger.info(f"saved to {save_path}")


@run_sarcasm_bench.command("openai", help="Run OpenAI processor")
@click.option("--model", type=str, default="gpt-4o", help="Model to use")
@click.option("--num-proc", type=int, default=-1, help="Number of processes to use")
def run_openai(
    model: str,
    num_proc: int,
):

    if num_proc <= 0:
        cpucount = os.cpu_count()
        if cpucount is None:
            cpucount = 16
        num_proc = int(cpucount * 0.8)
    logger.info(f"num_proc: {num_proc}")

    def processor(dataset, config_path, seed):
        dataset = dataset.map(
            partial(
                openai_requests_map_func,
                config_file_path=config_path,
                model=model,
                seed=seed,
            ),
            batched=True,
            num_proc=num_proc,
        )
        return dataset

    return processor


@run_sarcasm_bench.command("vllm", help="Run VLLM processor")
@click.option(
    "--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model to use"
)
@click.option("--num-proc", type=int, default=-1, help="Number of processes to use")
def run_vllm(
    model: str,
    num_proc: int,
):

    if num_proc <= 0:
        cpucount = os.cpu_count()
        if cpucount is None:
            cpucount = 16
        num_proc = int(cpucount * 0.8)
    logger.info(f"num_proc: {num_proc}")

    def processor(dataset, config_path, seed):
        dataset = dataset.map(
            partial(
                vllm_requests_map_func,
                config_file_path=config_path,
                model=model,
                seed=seed,
            ),
            batched=True,
            num_proc=num_proc,
        )
        return dataset

    return processor


if __name__ == "__main__":
    run_sarcasm_bench()
