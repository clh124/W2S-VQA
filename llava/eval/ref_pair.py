import argparse

import os
import json
from tqdm import tqdm
import numpy as np
import shortuuid
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

def wa5(logits):
    import numpy as np
    t=1
    logprobs = np.array([logits["superior"]/t, logits["better"]/t, logits["similar"]/t, logits["worse"]/t, logits["inferior"]/t])
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

        frame_idx1.extend([i for i in range(ii*round(vr.get_avg_fps()), (ii+1)*round(vr.get_avg_fps()), avg_fps)])
    total_frame_num = len(vr)-(len(vr)//round(vr.get_avg_fps())*round(vr.get_avg_fps()))
    avg_fps = round(vr.get_avg_fps() / video_fps)

    frame_idx1.extend([i for i in range((ii+1)*round(vr.get_avg_fps()), len(vr), avg_fps)])

    frames = vr.get_batch(frame_idx1).asnumpy()

    return [Image.fromarray(frames[i]) for i in range(len(frame_idx1))],frame_idx1


def load_motion_feature(image_id):
    motion_root = '/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/code/llm_pair_vqa_confidence_loss/llava/eval/anchor_videos/slowfast_feature/'
    motion_feat_list = []
    # find the max motion token
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

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # model.save_pretrained("llava-ov-chat-qwen2_slowfast_instruction_direct_1")
    os.makedirs(f"results/{args.model_path.split('/')[-1]}/", exist_ok=True)

    image_path =  "./llava/eval/anchor_videos/videos/"

    json_ = "./llava/eval/stadard.json"


    spearmanr1 = []
    personr1 = []
    matrix = np.zeros((5, 5))
    np.fill_diagonal(matrix, 0.5)
    print(matrix)

    inp = "Now you will receive two videos. The first video:\n <image><image>. The second video:\n <image><image>. Please watch these videos carefully, and then answer the following question: Comparing with the first video, how do you assess the quality of the second video?"
    with open(json_) as f:
        iqadata = json.load(f)
        
        for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):

            filename = llddata["img_path"]
            llddata["logits"] = defaultdict(float)
            cur_prompt = args.extra_prompt + inp
            print(cur_prompt)
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], cur_prompt)
            conv.append_message(conv.roles[1], "The quality of the second video is")
            prompt = conv.get_prompt()

            input_ids = preprocess_qwen([cur_prompt,{'from': 'gpt','value': "The quality of the second video is"}], tokenizer, has_image=True).cuda()

            img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

            image_tensors = []
            transformations_test = transforms.Compose(
                [transforms.Resize([224, 224]), transforms.ToTensor(), \
                    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

            videos = []
            frame_indices = []
            motion_feats = []
            for single_video_file in filename:
                slowfast_feature = load_motion_feature(single_video_file[:-4])
                image,frame_idx = load_video(image_path + single_video_file)
                videos.append(image)
                frame_indices.append(frame_idx)
                motion_feats.append(slowfast_feature)
            
            images = []
            for image in videos:
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                images.append(image_tensor)

            for image_file in image:
                image_tensor1 = transformations_test(image_file)
                image_tensors.append(image_tensor1)
            image_tensors = torch.stack(image_tensors)

            image_tensors = [
                (
                        [motion_feat.half().cuda()],  
                        [img.half().cuda()]              
                )
                for idx, (img, motion_feat) in enumerate(zip(images, motion_feats))
            ]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


            with torch.inference_mode():
                output_logits = model(input_ids,
                                    images=image_tensors,modalities=['video', 'video'])['logits'][:, -3]
            skip_token_id = [198, 13, 151643, 151644, 151645]

            llddata["logits"]["superior"] += output_logits.mean(0)[16353].item()
            llddata["logits"]["better"] += output_logits.mean(0)[2664].item()
            llddata["logits"]["similar"] += output_logits.mean(0)[4428].item()
            llddata["logits"]["worse"] += output_logits.mean(0)[10960].item()
            llddata["logits"]["inferior"] += output_logits.mean(0)[37179].item()
            print(llddata["logits"])

            llddata["pr_score"] = wa5(llddata["logits"])


            if i==0:
                matrix[1,0]=llddata["pr_score"]
                matrix[0,1]=1-llddata["pr_score"]
            if i==1:
                matrix[2,0]=llddata["pr_score"]
                matrix[0,2]=1-llddata["pr_score"]                      
            if i==2:
                matrix[3,0]=llddata["pr_score"]
                matrix[0,3]=1-llddata["pr_score"]
            if i==3:
                matrix[4,0]=llddata["pr_score"]
                matrix[0,4]=1-llddata["pr_score"]
            if i==4:
                matrix[2,1]=llddata["pr_score"]
                matrix[1,2]=1-llddata["pr_score"]                       
            if i==5:
                matrix[3,1]=llddata["pr_score"]
                matrix[1,3]=1-llddata["pr_score"]
            if i==6:
                matrix[4,1]=llddata["pr_score"]
                matrix[1,4]=1-llddata["pr_score"]
            if i==7:
                matrix[3,2]=llddata["pr_score"]
                matrix[2,3]=1-llddata["pr_score"]
            if i==8:
                matrix[4,2]=llddata["pr_score"]
                matrix[2,4]=1-llddata["pr_score"]  
            if i==9:
                matrix[4,3]=llddata["pr_score"]
                matrix[3,4]=1-llddata["pr_score"]           

            print(matrix)
            print("np.array(")
            print(np.array2string(matrix, separator=', ', formatter={'float_kind': lambda x: f"{x:.8e}"}))
            print(", dtype=np.float32)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/shared-storage-user/ailab-pceval/zhuxiangyang/caolinhan/weights/llava_qwen_stage2_no_lable_refinement/")
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
