import argparse
import csv
import os
import json
from tqdm import tqdm
import numpy as np
import shortuuid
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from torchvision import transforms
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re
from collections import defaultdict
from PIL import Image
import math
from scipy.stats import spearmanr, pearsonr
from ref_pair import cal_anchor_matrix

rng = np.random.default_rng()

def norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


def optimize_score_map_pytorch_cuda(c, seed=0, original_seed=20020, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    c = torch.tensor(c, dtype=torch.float32, device=device, requires_grad=False)
    initial_scores = torch.rand(c.shape[0], device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([initial_scores], lr=0.1)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        sum_log_diff = torch.sum(c * torch.log(torch.maximum(torch.sigmoid(initial_scores[:, None] - initial_scores), torch.tensor(1e-6, device=device))))
        sum_squares = torch.sum(initial_scores ** 2) / 2

        loss = -(sum_log_diff - sum_squares)
        loss.backward()
        optimizer.step()
    
    optimized_scores = initial_scores.detach().cpu().numpy()
    min_score, max_score = np.min(optimized_scores), np.max(optimized_scores)
    
    # Scale scores to 0-100
    scaled_scores = 100 * (optimized_scores - min_score) / (max_score - min_score)
    
    # Reset the seed
    np.random.seed(original_seed)
    return scaled_scores[-1]


def softmax(logits):
    # exp_logits = np.exp(logits - np.max(logits))
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs
    # return exp_logits / exp_logits.sum()

def update_matrices(preference_matrix, scores, indices):
    n = preference_matrix.shape[0]
    new_row = np.zeros((1, n))
    new_col = np.zeros((n + 1, 1))
    new_row[0, indices] = scores
    new_col[indices, 0] = 1-scores  # Assuming symmetric preference for simplicity
    preference_matrix = np.vstack([preference_matrix, new_row])
    preference_matrix = np.hstack([preference_matrix, new_col])
    preference_matrix[n, n] = 0.5
    return preference_matrix

def wa5(logits):
    import numpy as np
    logprobs = np.array([logits["superior"], logits["better"], logits["similar"], logits["worse"], logits["inferior"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1,0.75,0.5,0.25,0.]))

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def load_video(video_file):
    from decord import VideoReader,cpu
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)

    frame_idx1 = []
    video_fps=1
    for ii in range(len(vr)//round(vr.get_avg_fps())):
        total_frame_num = round(vr.get_avg_fps())
        avg_fps = round(vr.get_avg_fps() / video_fps)
        # total_frame_num=len(vr)//avg_fps*avg_fps
        frame_idx1.extend([i for i in range(ii*round(vr.get_avg_fps()), (ii+1)*round(vr.get_avg_fps()), avg_fps)])
    total_frame_num = len(vr)-(len(vr)//round(vr.get_avg_fps())*round(vr.get_avg_fps()))
    avg_fps = round(vr.get_avg_fps() / video_fps)
    # total_frame_num=len(vr)//avg_fps*avg_fps
    frame_idx1.extend([i for i in range((ii+1)*round(vr.get_avg_fps()), len(vr), avg_fps)])

    frames = vr.get_batch(frame_idx1).asnumpy()

    return [Image.fromarray(frames[i]) for i in range(len(frame_idx1))],frame_idx1
    # return frame_idx,len(frame_idx)/video_fps

def load_motion_feature(image_id, motion_root):
    # motion_root = '/root/autodl-tmp/LSVQ/LSVQ_Train_SlowFast_feature/'
    motion_feat_list = []
    # find the max motion token
    # print(image_id, motion_root)
    max_motion_idx = sorted(os.listdir(os.path.join(motion_root, image_id)))[-1].split('_')[1]
    # print("max_motion_idx", max_motion_idx)
    for img_index in range(int(max_motion_idx)+1):
        fast_mo_feat = os.path.join(motion_root, image_id,f"feature_{img_index}_fast_feature.npy")

        motion_feat_per_img = torch.from_numpy(np.load(fast_mo_feat)).squeeze()
        # print(motion_feat_per_img.shape)

        motion_feat_list.append(motion_feat_per_img.unsqueeze(0))

    motion_feat = torch.cat(motion_feat_list, 0)
    # print(motion_feat.shape)

    return motion_feat


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    # if roles[source[0]["from"]] != roles["human"]:
    #     source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        if j==0:
            role = "<|im_start|>user"
        else:
            role = "<|im_start|>assistant"
        if has_image and sentence is not None and "<image>" in sentence:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence))
            texts = sentence.split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX]
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            print(f"Parameter {name} is on meta. Moving to GPU.")


    os.makedirs(f"results/{args.model_path.split('/')[-1]}/", exist_ok=True)


    anchor_matrix = cal_anchor_matrix(args)
    print("anchor_matrix:", anchor_matrix)

    anchor_intervals = 5#16
    num_anchor_image_per_interval = 1
    num_anchor_image = anchor_intervals * num_anchor_image_per_interval
    anchor_indices = np.arange(0, num_anchor_image)

    image_paths = [
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/LIVE_VQC/Video/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/KoNViD_1k/KoNViD_1k_videos/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/youtube_ugc/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/live_yt_gaming/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/cgvds/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/LSVQ/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/LSVQ/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/kvq/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/waterloo_ivc_4k/",
            "/mnt/shared-storage-user/zhuxiangyang/tos/wenfarong/caolinhan/data/test_data/live_yt_hfr/",
        ]
    
    motion_features = [
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/slowfast_feature_live_vqc/",
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/slowfast_feature_konvid_1k/",
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/slowfast_feature_youtube_ugc/",
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/live_yt_gaming_slowfast_feature/",
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/cgvds_slowfast_feature/",      
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/LSVQ_Train_SlowFast_feature/",
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/LSVQ_Test_SlowFast_feature/",
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/kvq_slowfast_feature/",
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/waterloo_slowfast_feature/",
        "/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/data/slowfast_feature/live_hfr_slowfast_feature/",
    ]
    anchor_image_path = "./llava/eval/anchor_videos/videos/"
    # anchor_image_path = "/data2/caolinhan/video_database/train_70w/videos/"
    anchor_motion_path = "./llava/eval/anchor_videos/slowfast_feature/"

    json_prefix = './llava/eval/pair_json_path/'
    
    jsons = [
            json_prefix + "LIVE-VQC_total_ds_score.json",
            json_prefix + "Konvid-1k_total_ds_score.json",
            json_prefix + "youtube_ugc_total.json",
            json_prefix + "LIVE-YT-Gaming_total_score.json",
            json_prefix+ "CGVDS_total_score.json",
            json_prefix + "LSVQ_whole_test_ds_score.json",
            json_prefix + "LSVQ_whole_test_1080p_ds_score.json",
            json_prefix + "kvq_train_score.json",
            json_prefix + "Waterloo_IVC_4K_total_score2.json",
            json_prefix + "live_hfr_total_score.json",
        ]

    # 存放结果的 CSV 文件的文件夹
    csv_output_folder = "./results/"  # 修改为你希望存放 CSV 文件的文件夹

    # 确保目标文件夹存在
    os.makedirs(csv_output_folder, exist_ok=True)

    anchor_json = "./llava/eval/anchor.json"
    with open(anchor_json) as f:
        anchor_vqadata = json.load(f)
    anchor_tensors = []
    anchor_motion_feats = []

    for anchordata in anchor_vqadata:
        anchorname = anchor_image_path + anchordata["img_path"]
        image1, _ = load_video(anchorname)
        image_tensor1 = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'].half().cuda()
        motion_feat1 = load_motion_feature(anchordata["img_path"][:-4], anchor_motion_path).half().cuda()

        anchor_tensors.append(image_tensor1)
        anchor_motion_feats.append(motion_feat1)

    print(f"Preloaded {len(anchor_tensors)} anchor videos successfully.")


    inp = "Now you will receive two videos. The first video:\n <image><image>. The second video:\n <image><image>. Please watch these videos carefully, and then answer the following question: Comparing with the first video, how do you assess the quality of the second video?"
    for image_path, json_, motion_feature in zip(image_paths, jsons, motion_features):
        print(image_path, json_)
        with open(json_) as f:
            iqadata = json.load(f)
        #     with open(json_) as f:
        #         iqadata = json.load(f)
        csv_filename = os.path.basename(json_).replace(".json", ".csv")  # 仅保留文件名，去掉路径
        csv_output_path = os.path.join(csv_output_folder, csv_filename)  # 目标文件路径
    
        with open(csv_output_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["filename", "pred_score", "gt_score"])  # 写入表头

            gt_scores = []  
            pre_soft_score = []
            for i, llddata in enumerate(tqdm(iqadata["annotations"], desc="Evaluating [{}]".format(json_.split("/")[-1]))):

                # try:
                if True:

                    llddata["image_id"] = llddata["image_id"] if llddata["image_id"].endswith('.mp4') else llddata["image_id"] + '.mp4'

                    filename = image_path + llddata["image_id"]
                    print("filename", filename)
                    gt_score = llddata["score"]
                    probabilities = []
                    slowfast_feature2 = load_motion_feature(llddata["image_id"][:-4], motion_feature)
                    image2,_ = load_video(filename)
                    image_tensor2 = image_processor.preprocess(image2, return_tensors='pt')['pixel_values']

                    for anchor_tensor, anchor_motion_feat in zip(anchor_tensors, anchor_motion_feats):
                            images = [anchor_tensor, image_tensor2]
                            motion_feats = [anchor_motion_feat, slowfast_feature2]
                            llddata["logits"] = defaultdict(float)
                            cur_prompt = args.extra_prompt + inp
                            conv = conv_templates[args.conv_mode].copy()
                            conv.append_message(conv.roles[0], cur_prompt)
                            conv.append_message(conv.roles[1], "The quality of the second video is")

                            input_ids = preprocess_qwen([cur_prompt,{'from': 'gpt','value': "The quality of the second video is"}], tokenizer, has_image=True).cuda()

                            image_tensors = [
                            (
                                    [motion_feat.half().cuda()],  # 每个视频的图像，按帧数裁剪
                                    [img.half().cuda()]              # 每个视频对应的帧索引
                            )
                            for idx, (img, motion_feat) in enumerate(zip(images, motion_feats))
                            ]


                            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                            keywords = [stop_str]
                            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                            with torch.inference_mode():
                                output_logits = model(input_ids,
                                                    images=image_tensors,modalities=['video', 'video'])[0][:, -3]

                            skip_token_id = [198, 13, 151643, 151644, 151645]

                            llddata["logits"]["superior"] += output_logits.mean(0)[16353].item()
                            llddata["logits"]["better"] += output_logits.mean(0)[2664].item()
                            llddata["logits"]["similar"] += output_logits.mean(0)[4428].item()
                            llddata["logits"]["worse"] += output_logits.mean(0)[10960].item()
                            llddata["logits"]["inferior"] += output_logits.mean(0)[37179].item()
                            # print(llddata["logits"])
                            comparison = llddata
                            t = 5
                            logits = np.array([comparison["logits"]["inferior"]/t, comparison["logits"]["worse"]/t, comparison["logits"]["similar"]/t, comparison["logits"]["better"]/t, comparison["logits"]["superior"]/t])
                            probability = softmax(logits)
                            preference = np.inner(probability, np.array([0,0.25,0.5,0.75,1.]))
                            # print("preference", preference)
                            probabilities.append(preference)


                            llddata["pr_score"] = wa5(llddata["logits"])

                            # print(llddata["pr_score"], llddata["score"])
                            # with torch.inference_mode():
                            #     output_ids = model.generate(
                            #         input_ids,
                            #         images=image_tensors,
                            #         do_sample=True if args.temperature > 0 else False,
                            #         temperature=args.temperature,
                            #         top_p=args.top_p,
                            #         num_beams=args.num_beams,
                            #         # no_repeat_ngram_size=3,
                            #         max_new_tokens=1024,
                            #         use_cache=True)
                            # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                            # # outputs = outputs.strip()
                            # print("outputs", outputs)
                                
                    updated_matrix = update_matrices(anchor_matrix, np.array(probabilities), anchor_indices)
                    # print("Preference matrix construction complete.")
                    pred_score = optimize_score_map_pytorch_cuda(updated_matrix, seed=0, original_seed=20020, num_iterations=50)

                    print("soft_map_result_score 100 is: ", pred_score)
                    
                    pre_soft_score.append(pred_score)
                    gt_scores.append(float(gt_score))
                    csv_writer.writerow([filename, pred_score, gt_score])
                    if i>0:

                        print("Spearmanr", spearmanr(pre_soft_score, gt_scores)[0], "Pearson", pearsonr(pre_soft_score, gt_scores)[0])
                    

                else:
                # except:
                    print(filename, "not exists!")
            

    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/weights/llava_qwen_stage1")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)

