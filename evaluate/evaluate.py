"""HPDv3++ pairwise preference evaluation for HPSv3++.

Evaluates a single HPSv3++ checkpoint on the HPDv3++ pairwise test sets
(datasets/test/test_aes.json and datasets/test/test_tf.json). Each test item is a
preference pair {path1, path2, prompt} where path1 is the preferred image.
The reward model scores both images and a pair is counted correct when the
preferred image receives the higher reward. Accuracy is reported across all
pairs. Evaluation is sharded across the available GPUs with one worker process
per shard.
"""
import os
import json
import math
import multiprocessing as mp
from multiprocessing import Process, Queue

import torch
import fire
import prettytable
from tqdm import tqdm

from hpsv3.inference import HPSv3RewardInferencer


def worker_process(process_id, data_chunk, config_path, checkpoint_path, batch_size, result_queue, img_root):
    num_gpus = torch.cuda.device_count()
    device = f"cuda:{process_id % num_gpus}" if num_gpus > 0 else "cpu"

    print(f"Process {process_id} starting with device {device}, processing {len(data_chunk)} items")

    inferencer = HPSv3RewardInferencer(config_path, checkpoint_path, device=device)

    process_correct = 0
    process_equal = 0
    process_results = []

    def _abs(p):
        p = str(p)
        if os.path.isabs(p):
            return p
        # Test JSON stores relative paths such as images/...; join them under img_root (the dataset root).
        return os.path.join(img_root, p)

    for batch_start in tqdm(
        range(0, len(data_chunk), batch_size),
        total=(len(data_chunk) + batch_size - 1) // batch_size,
        desc=f"Process {process_id}",
    ):
        batch_end = min(batch_start + batch_size, len(data_chunk))
        batch_info = data_chunk[batch_start:batch_end]

        prompts = [info["prompt"] for info in batch_info]
        image_paths_1 = [_abs(info["path1"]) for info in batch_info]
        image_paths_2 = [_abs(info["path2"]) for info in batch_info]

        missing_1 = [p for p in image_paths_1 if not os.path.exists(p)]
        missing_2 = [p for p in image_paths_2 if not os.path.exists(p)]
        if missing_1 or missing_2:
            msg = []
            if missing_1:
                msg.append(f"missing path1 (show up to 5): {missing_1[:5]}")
            if missing_2:
                msg.append(f"missing path2 (show up to 5): {missing_2[:5]}")
            raise FileNotFoundError(" | ".join(msg))

        with torch.no_grad():
            rewards_1 = inferencer.reward(image_paths=image_paths_1, prompts=prompts)
            rewards_2 = inferencer.reward(image_paths=image_paths_2, prompts=prompts)

        for i in range(len(batch_info)):
            info = batch_info[i]
            if rewards_1.ndim == 2:
                reward_1, reward_2 = rewards_1[i][0].item(), rewards_2[i][0].item()
            else:
                reward_1, reward_2 = rewards_1[i].item(), rewards_2[i].item()

            process_results.append({
                "reward_1": reward_1,
                "reward_2": reward_2,
                "correct": reward_1 > reward_2,
                "equal": reward_1 == reward_2,
                "info": info,
            })

            if reward_1 > reward_2:
                process_correct += 1
            if reward_1 == reward_2:
                process_equal += 1

    result_queue.put({
        "process_id": process_id,
        "correct": process_correct,
        "equal": process_equal,
        "total": len(data_chunk),
        "results": process_results,
    })

    print(
        f"Process {process_id} completed: {process_correct}/{len(data_chunk)} correct, "
        f"{process_equal}/{len(data_chunk)} equal"
    )


def main(test_json, config_path=None, batch_size=8, num_processes=8, checkpoint_path=None,
         mode="pair", img_root=None, per_item_json=None):
    assert mode == "pair", "Only pairwise preference evaluation is supported."
    assert checkpoint_path is not None, "Checkpoint path must be provided for inference"
    assert img_root is not None, "img_root (the dataset root, e.g. datasets) must be provided"

    mp.set_start_method("spawn", force=True)

    info_list = json.load(open(test_json, "r"))

    print("[INFO] test_json =", test_json)
    print(f"[INFO] total items to process: {len(info_list)}")
    print(f"[INFO] img_root = {img_root}")

    chunk_size = math.ceil(len(info_list) / num_processes)
    data_chunks = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(info_list))
        if start_idx < len(info_list):
            chunk = info_list[start_idx:end_idx]
            data_chunks.append(chunk)
            print(f"Process {i}: {len(chunk)} items (indices {start_idx}-{end_idx-1})")

    actual_processes = len(data_chunks)
    print(f"Using {actual_processes} processes")

    result_queue = Queue()
    processes = []

    print("Starting processes...")
    for i in range(actual_processes):
        p = Process(
            target=worker_process,
            args=(i, data_chunks[i], config_path, checkpoint_path, batch_size, result_queue, img_root),
        )
        p.start()
        processes.append(p)

    all_results = []
    total_correct = 0
    total_equal = 0
    total_items = 0

    print("Waiting for processes to complete...")
    for _ in range(actual_processes):
        result = result_queue.get()
        all_results.append(result)
        total_correct += result["correct"]
        total_equal += result["equal"]
        total_items += result["total"]
        print(
            f"Process {result['process_id']} finished: {result['correct']}/{result['total']} correct, "
            f"{result['equal']}/{result['total']} equal"
        )

    for p in processes:
        p.join()

    incorrect = total_items - total_correct - total_equal
    accuracy_percent = 100 * total_correct / total_items if total_items > 0 else 0.0

    table = prettytable.PrettyTable()
    table.field_names = ["Total Items", "Correct", "Equal", "Incorrect", "Accuracy (%)"]
    table.add_row([total_items, total_correct, total_equal, incorrect, f"{accuracy_percent:.2f}"])

    if per_item_json:
        flat = []
        for r in all_results:
            for item in r.get("results", []):
                info = item.get("info", {})
                flat.append({
                    "path1": info.get("path1"),
                    "path2": info.get("path2"),
                    "prompt": info.get("prompt"),
                    "reward_1": item.get("reward_1"),
                    "reward_2": item.get("reward_2"),
                    "correct": item.get("correct"),
                    "equal": item.get("equal"),
                })
        os.makedirs(os.path.dirname(per_item_json) or ".", exist_ok=True)
        with open(per_item_json, "w") as f:
            json.dump(flat, f, ensure_ascii=False)
        print(f"Saved per-item results: {per_item_json} ({len(flat)} items)")

    print(table)


if __name__ == "__main__":
    fire.Fire(main)
