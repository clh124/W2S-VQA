import argparse
import csv
import json
import os
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict

import numpy as np
import torch
import transformers
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


PAIR_PROMPT = (
    "Now you will receive two videos. The first video:\n <image><image>. "
    "The second video:\n <image><image>. Please watch these videos carefully, "
    "and then answer the following question: Comparing with the first video, "
    "how do you assess the quality of the second video?"
)
ANSWER_PREFIX = "The quality of the second video is"
QUALITY_TOKEN_IDS = {
    "superior": 16353,
    "better": 2664,
    "similar": 4428,
    "worse": 10960,
    "inferior": 37179,
}
ANCHOR_IMAGE_PATH = "llava/eval/anchor_videos/videos/"
ANCHOR_MOTION_PATH = "llava/eval/anchor_videos/slowfast_feature/"
ANCHOR_JSON = "llava/eval/anchor.json"


def optimize_score_map_pytorch_cuda(c, seed=0, original_seed=20020, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    c = torch.tensor(c, dtype=torch.float32, device=device, requires_grad=False)
    initial_scores = torch.rand(c.shape[0], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([initial_scores], lr=0.1)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        pair_prob = torch.maximum(
            torch.sigmoid(initial_scores[:, None] - initial_scores),
            torch.tensor(1e-6, device=device),
        )
        sum_log_diff = torch.sum(c * torch.log(pair_prob))
        sum_squares = torch.sum(initial_scores**2) / 2
        loss = -(sum_log_diff - sum_squares)
        loss.backward()
        optimizer.step()

    optimized_scores = initial_scores.detach().cpu().numpy()
    min_score, max_score = np.min(optimized_scores), np.max(optimized_scores)
    scaled_scores = 100 * (optimized_scores - min_score) / (max_score - min_score)

    np.random.seed(original_seed)
    return scaled_scores[-1]


def softmax(logits):
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs


def update_matrices(preference_matrix, scores, indices):
    n = preference_matrix.shape[0]
    new_row = np.zeros((1, n))
    new_col = np.zeros((n + 1, 1))
    new_row[0, indices] = scores
    new_col[indices, 0] = 1 - scores
    preference_matrix = np.vstack([preference_matrix, new_row])
    preference_matrix = np.hstack([preference_matrix, new_col])
    preference_matrix[n, n] = 0.5
    return preference_matrix


def wa5(logits):
    logprobs = np.array(
        [
            logits["superior"],
            logits["better"],
            logits["similar"],
            logits["worse"],
            logits["inferior"],
        ]
    )
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1, 0.75, 0.5, 0.25, 0.0]))


def load_video(video_file, video_fps):
    from decord import VideoReader, cpu

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    frame_idx = []
    video_fps = 1

    for ii in range(len(vr) // round(vr.get_avg_fps())):
        avg_fps = round(vr.get_avg_fps() / video_fps)
        frame_idx.extend(
            [i for i in range(ii * round(vr.get_avg_fps()), (ii + 1) * round(vr.get_avg_fps()), avg_fps)]
        )

    avg_fps = round(vr.get_avg_fps() / video_fps)
    start_idx = (ii + 1) * round(vr.get_avg_fps()) if len(vr) // round(vr.get_avg_fps()) else 0
    frame_idx.extend([i for i in range(start_idx, len(vr), avg_fps)])

    frames = vr.get_batch(frame_idx).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(len(frame_idx))], frame_idx


def load_motion_feature(image_id, motion_root):
    motion_feat_list = []
    feature_dir = os.path.join(motion_root, image_id)
    max_motion_idx = sorted(os.listdir(feature_dir))[-1].split("_")[1]

    for img_index in range(int(max_motion_idx) + 1):
        fast_mo_feat = os.path.join(feature_dir, f"feature_{img_index}_fast_feature.npy")
        motion_feat_per_img = torch.from_numpy(np.load(fast_mo_feat)).squeeze()
        motion_feat_list.append(motion_feat_per_img.unsqueeze(0))

    return torch.cat(motion_feat_list, 0)


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens

    for j, sentence in enumerate(sources):
        role = "<|im_start|>user" if j == 0 else "<|im_start|>assistant"
        if has_image and sentence is not None and "<image>" in sentence:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence))
            texts = sentence.split("<image>")
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX]
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX for i in _input_id]) == num_image
        elif sentence["value"] is None:
            _input_id = tokenizer(role).input_ids + nl_tokens
        else:
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens

        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        else:
            role_len = len(tokenizer(role).input_ids)
            _target = [im_start] + [IGNORE_INDEX] * role_len + _input_id[role_len + 1 : -2] + [im_end] + nl_tokens
        target += _target

    return torch.tensor([input_id], dtype=torch.long)


def model_logits(model, input_ids, image_tensors):
    output = model(input_ids, images=image_tensors, modalities=["video", "video"])
    logits = output["logits"] if isinstance(output, dict) else output[0]
    return logits[:, -3]


def infer_pair_preference(args, model, tokenizer, first_tensor, first_motion, second_tensor, second_motion, temperature=5):
    input_ids = preprocess_qwen(
        [args.extra_prompt + PAIR_PROMPT, {"from": "gpt", "value": ANSWER_PREFIX}],
        tokenizer,
        has_image=True,
    ).cuda()

    image_tensors = [
        ([first_motion.half().cuda()], [first_tensor.half().cuda()]),
        ([second_motion.half().cuda()], [second_tensor.half().cuda()]),
    ]

    with torch.inference_mode():
        output_logits = model_logits(model, input_ids, image_tensors)

    logits = defaultdict(float)
    for label, token_id in QUALITY_TOKEN_IDS.items():
        logits[label] += output_logits.mean(0)[token_id].item()

    scaled_logits = np.array(
        [
            logits["inferior"] / temperature,
            logits["worse"] / temperature,
            logits["similar"] / temperature,
            logits["better"] / temperature,
            logits["superior"] / temperature,
        ]
    )
    probability = softmax(scaled_logits)
    preference = np.inner(probability, np.array([0, 0.25, 0.5, 0.75, 1.0]))
    return preference, wa5(logits), logits


def cache_video(image_processor, video_root, motion_root, filename):
    video_name = filename if filename.endswith(".mp4") else filename + ".mp4"
    images, _ = load_video(os.path.join(video_root, video_name), 24)
    image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    motion_feat = load_motion_feature(video_name[:-4], motion_root)
    return image_tensor, motion_feat


def load_anchor_cache(image_processor, anchors, video_root, motion_root):
    anchor_tensors = []
    anchor_motion_feats = []
    anchor_name_to_index = {}
    for idx, item in enumerate(anchors):
        image_tensor, motion_feat = cache_video(image_processor, video_root, motion_root, item["img_path"])
        anchor_tensors.append(image_tensor.half().cuda())
        anchor_motion_feats.append(motion_feat.half().cuda())

        anchor_name_to_index[item["id"]] = idx
        name = os.path.basename(item["img_path"])
        anchor_name_to_index[name] = idx
        anchor_name_to_index[os.path.splitext(name)[0]] = idx

    print(f"Preloaded {len(anchor_tensors)} anchor videos successfully.")
    return anchor_tensors, anchor_motion_feats, anchor_name_to_index


def build_anchor_matrix(args, model, tokenizer, anchor_pairs, anchor_tensors, anchor_motion_feats, anchor_name_to_index):
    matrix = np.zeros((len(anchor_tensors), len(anchor_tensors)), dtype=np.float32)
    np.fill_diagonal(matrix, 0.5)
    fallback_pairs = list(combinations(range(len(anchor_tensors)), 2))

    for pair_idx, item in enumerate(tqdm(anchor_pairs, desc="Building anchor matrix")):
        first_idx, second_idx = resolve_anchor_pair(item, pair_idx, fallback_pairs, anchor_name_to_index)
        preference, pr_score, logits = infer_pair_preference(
            args,
            model,
            tokenizer,
            anchor_tensors[first_idx],
            anchor_motion_feats[first_idx],
            anchor_tensors[second_idx],
            anchor_motion_feats[second_idx],
            temperature=1,
        )
        matrix[second_idx, first_idx] = pr_score
        matrix[first_idx, second_idx] = 1 - pr_score
        # print(logits)
        # print("anchor preference", preference, "anchor pr_score", pr_score)
        # print(matrix)

    return matrix


def resolve_anchor_pair(item, pair_idx, fallback_pairs, anchor_name_to_index):
    anchor_ids = item.get("anchors", [])
    if isinstance(anchor_ids, list) and len(anchor_ids) >= 2:
        first_idx = anchor_name_to_index.get(anchor_ids[0])
        second_idx = anchor_name_to_index.get(anchor_ids[1])
        if first_idx is not None and second_idx is not None:
            return first_idx, second_idx

    img_paths = item.get("img_path", [])
    if isinstance(img_paths, list) and len(img_paths) >= 2:
        first_name = os.path.basename(img_paths[0])
        second_name = os.path.basename(img_paths[1])
        first_idx = anchor_name_to_index.get(first_name, anchor_name_to_index.get(os.path.splitext(first_name)[0]))
        second_idx = anchor_name_to_index.get(second_name, anchor_name_to_index.get(os.path.splitext(second_name)[0]))
        if first_idx is not None and second_idx is not None:
            return first_idx, second_idx

    if pair_idx >= len(fallback_pairs):
        raise ValueError(f"Cannot resolve anchor pair at index {pair_idx}: {item}")
    return fallback_pairs[pair_idx]


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    for name, param in model.named_parameters():
        if param.device.type == "meta":
            print(f"Parameter {name} is on meta. Moving to GPU.")

    os.makedirs(f"results/{args.model_path.split('/')[-1]}/", exist_ok=True)
    os.makedirs(args.csv_output_folder, exist_ok=True)

    with open(ANCHOR_JSON) as f:
        anchor_data = json.load(f)

    anchor_tensors, anchor_motion_feats, anchor_name_to_index = load_anchor_cache(
        image_processor,
        anchor_data["anchors"],
        ANCHOR_IMAGE_PATH,
        ANCHOR_MOTION_PATH,
    )
    anchor_matrix = build_anchor_matrix(
        args,
        model,
        tokenizer,
        anchor_data["pairs"],
        anchor_tensors,
        anchor_motion_feats,
        anchor_name_to_index,
    )
    anchor_indices = np.arange(0, len(anchor_tensors))

    for image_path, json_path, motion_feature in zip(args.image_paths, args.jsons, args.motion_features):
        print(image_path, json_path)
        with open(json_path) as f:
            iqadata = json.load(f)

        csv_filename = os.path.basename(json_path).replace(".json", ".csv")
        csv_output_path = os.path.join(args.csv_output_folder, csv_filename)

        with open(csv_output_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["filename", "pred_score", "gt_score"])

            gt_scores = []
            pre_soft_score = []
            for i, llddata in enumerate(tqdm(iqadata["annotations"], desc=f"Evaluating [{os.path.basename(json_path)}]")):
                try:
                    image_id = llddata["image_id"] if llddata["image_id"].endswith(".mp4") else llddata["image_id"] + ".mp4"
                    filename = os.path.join(image_path, image_id)
                    gt_score = llddata["score"]
                    image_tensor2, slowfast_feature2 = cache_video(image_processor, image_path, motion_feature, image_id)

                    probabilities = []
                    for anchor_tensor, anchor_motion_feat in zip(anchor_tensors, anchor_motion_feats):
                        preference, pr_score, logits = infer_pair_preference(
                            args,
                            model,
                            tokenizer,
                            anchor_tensor,
                            anchor_motion_feat,
                            image_tensor2,
                            slowfast_feature2,
                        )
                        probabilities.append(preference)
                        # print(logits)
                        # print("preference", preference)
                        # print(pr_score, gt_score)

                    updated_matrix = update_matrices(anchor_matrix, np.array(probabilities), anchor_indices)
                    pred_score = optimize_score_map_pytorch_cuda(updated_matrix, seed=0, original_seed=20020, num_iterations=50)

                    print("soft_map_result_score 100 is: ", pred_score)

                    pre_soft_score.append(pred_score)
                    gt_scores.append(float(gt_score))
                    csv_writer.writerow([filename, pred_score, gt_score])
                    if i > 0:
                        print("Spearmanr", spearmanr(pre_soft_score, gt_scores)[0], "Pearson", pearsonr(pre_soft_score, gt_scores)[0])
                except Exception as exc:
                    print(filename, "failed:", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/dev/shm/llava_qwen_stage3_only_stage2_labeled_add_adapted_confidence_loss/",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--extra-prompt", type=str, default="")

    parser.add_argument(
        "--csv-output-folder",
        type=str,
        default="/mnt/shared-storage-user/zhuxiangyang/tos/caolinhan/code/llm_pair_vqa_confidence_loss/results/test_stage2_no_label_refinement/",
    )
    parser.add_argument(
        "--image-paths",
        nargs="+",
        default=[
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/LIVE_VQC/Video/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/waterloo_ivc_4k/",
        ],
    )
    parser.add_argument(
        "--motion-features",
        nargs="+",
        default=[
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/data/caolinhan_data/video_database/train_30w/slowfast_feature/slowfast_feature_live_vqc/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/data/caolinhan_data/video_database/train_30w/slowfast_feature/waterloo_slowfast_feature/",
        ],
    )
    parser.add_argument(
        "--jsons",
        nargs="+",
        default=[
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/pair_json_path/LIVE-VQC_total_ds_score.json",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/pair_json_path/Waterloo_IVC_4K_total_score2.json",
        ],
    )
    args = parser.parse_args()

    if not (len(args.image_paths) == len(args.motion_features) == len(args.jsons)):
        raise ValueError("--image-paths, --motion-features and --jsons must have the same length.")

    eval_model(args)
