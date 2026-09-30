import os
import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm

def compute_clip_score(image_path, text, model, preprocess, device):
    # 加载图片
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        # 提取特征
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # 归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 余弦相似度 = CLIP-Score
        score = (image_features @ text_features.T).item()

    return score

def evaluate_folder(folder_path, text, output_csv="clip_results.csv", model_name="ViT-L/14"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    results = []
    files = sorted(os.listdir(folder_path))

    for f in tqdm(files, desc="Processing images"):
        img_path = os.path.join(folder_path, f)
        if not os.path.isfile(img_path):
            continue

        try:
            score = compute_clip_score(img_path, text, model, preprocess, device)
            results.append({"filename": f, "CLIP-Score": score})
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # 保存结果
    df = pd.DataFrame(results)
    df.loc["Average"] = df.mean(numeric_only=True)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # 修改为你的路径和文本
    folder_path = "/home/huiluo/Data/FateZero/result/style/jeep_watercolor_260910-160959/sample/step_0_1_0"  # 存放图片的文件夹
    text = "watercolor painting of a silver jeep driving down a curvy road in the countryside."
    evaluate_folder(folder_path, text, output_csv="clip_scores.csv")