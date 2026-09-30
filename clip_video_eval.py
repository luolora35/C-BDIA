import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

def load_clip(model_name="ViT-B/32", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device

def get_image_features(image_path, model, preprocess, device):
    """提取图像特征并归一化"""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat

def get_text_features(text, model, device):
    """提取文本特征并归一化"""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat

def compute_temcon(frame_features):
    """相邻帧之间的 CLIP 特征相似度平均值"""
    sims = []
    for i in range(len(frame_features) - 1):
        sim = (frame_features[i] @ frame_features[i+1].T).item()
        sims.append(sim)
    return sum(sims) / len(sims) if sims else 0.0

def compute_framacc(frame_features, src_text_feat, tgt_text_feat):
    """逐帧判断是否更接近 target prompt"""
    correct = 0
    for feat in frame_features:
        sim_tgt = (feat @ tgt_text_feat.T).item()
        sim_src = (feat @ src_text_feat.T).item()
        if sim_tgt > sim_src:
            correct += 1
    return correct / len(frame_features) if frame_features else 0.0

if __name__ == "__main__":
    # ----------------------------
    # 输入设置
    # ----------------------------
    gen_frames_folder = "/data/huiluo/C-BDIA/result/style/DDIM40/sample/step_0_1_0"   # 生成后视频的帧
    source_prompt = "a silver jeep driving down a curvy road in the countryside"   # 原始 prompt
    target_prompt = "watercolor painting of a silver jeep driving down a curvy road in the countryside"      # 目标 prompt

    # ----------------------------
    # 加载模型
    # ----------------------------
    model, preprocess, device = load_clip("ViT-B/32")

    # 提取生成视频帧特征
    gen_files = sorted(os.listdir(gen_frames_folder))
    gen_features = []
    for f in tqdm(gen_files, desc="Extracting frame features"):
        img_path = os.path.join(gen_frames_folder, f)
        feat = get_image_features(img_path, model, preprocess, device)
        gen_features.append(feat)

    # 提取文本特征（用于 Fram-Acc）
    src_text_feat = get_text_features(source_prompt, model, device)
    tgt_text_feat = get_text_features(target_prompt, model, device)

    # ----------------------------
    # 计算指标
    # ----------------------------
    tem_con = compute_temcon(gen_features)  # 生成视频的时序一致性
    fram_acc = compute_framacc(gen_features, src_text_feat, tgt_text_feat)  # 编辑后帧的准确率

    print(f"Temporal Consistency (Tem-Con): {tem_con:.4f}")
    print(f"Frame-wise Editing Accuracy (Fram-Acc): {fram_acc:.4f}")
