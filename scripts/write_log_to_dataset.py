import datetime
import json
from functools import partial
from pathlib import Path
from typing import Any, Optional, cast

import click
import jsonlines
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
    f"write_log_to_dataset_{curent_datetime}.log",
    serialize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    enqueue=True,
)

from sarcbench.generator_map_functions import local_data_map_func


def extract_log(log_path: str) -> dict:
    all_data = {}

    reader = jsonlines.open(log_path)
    for obj in reader:
        level = obj["record"]["level"]["name"]
        if level != "SUCCESS":
            continue

        message: str = obj["record"]["message"]
        id_, _, data = message.split("-", maxsplit=2)
        response: dict[str, Any] = json.loads(data)

        model_name = response["model"].replace("/", "--")
        all_data[f"{id_}-{model_name}"] = response

    return all_data


@click.command()
@click.option("--dataset-path", "-d", type=str, help="Dataset to use")
@click.option("--dataset-name", "-n", type=str, required=False, help="Dataset name")
@click.option("--dataset-split", "-s", type=str, required=False, help="Dataset split")
@click.option("--log-path", "-l", type=str, help="Log file path")
@click.option("--model", "-m", type=str, multiple=True, help="Model name")
@click.option(
    "--output-path",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output path",
)
def write_log_to_dataset(
    dataset_path: str,
    dataset_name: Optional[str],
    dataset_split: Optional[str],
    log_path: str,
    model: list[str],
    output_path: Path,
) -> None:
    try:
        datasets = load_dataset(dataset_path, dataset_name, split=dataset_split)
        dataset_name = dataset_path.replace("/", "--")
    except ValueError:
        datasets = load_from_disk(dataset_path)
        dataset_name = dataset_path.split("/")[-1]

    if not isinstance(datasets, DatasetDict):
        datasets = {"default": datasets}

    all_response_data = extract_log(log_path)

    logger.debug(f"all_response_data: {list(all_response_data.keys())[:20]}")

    writed_datasets = {}
    for key, dataset in datasets.items():
        dataset = cast(Dataset, dataset)
        dataset = dataset.map(
            partial(local_data_map_func, local_data=all_response_data, models=model),
            batched=True,
            num_proc=1,  # No blocking requests to the server, so no need for multiple processes such that we can save memory
        )
        writed_datasets[key] = dataset

    output_path.mkdir(parents=True, exist_ok=True)
    for key, dataset in writed_datasets.items():
        output_path = output_path / f"{dataset_name}-{key}"
        dataset.save_to_disk(output_path)
        logger.info(f"saved to {output_path}")


if __name__ == "__main__":
    write_log_to_dataset()
