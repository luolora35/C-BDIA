import os
import cv2
import torch
import lpips
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 转 tensor 函数
def to_tensor(img):
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return img*2 - 1  # [0,1] -> [-1,1]

def calculate_metrics(img1_path, img2_path, lpips_fn):
    # 读取图像 (BGR->RGB)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise ValueError(f"Error reading {img1_path} or {img2_path}")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # PSNR
    psnr_val = psnr(img1, img2, data_range=1.0)

    # SSIM
    ssim_val = ssim(img1, img2, channel_axis=2, data_range=1.0)

    # LPIPS
    img1_t = to_tensor(img1)
    img2_t = to_tensor(img2)
    lpips_val = lpips_fn(img1_t, img2_t).item()

    return psnr_val, ssim_val, lpips_val


def evaluate_folder(folder_ref, folder_test, output_csv="results.csv"):
    # LPIPS 网络
    lpips_fn = lpips.LPIPS(net='alex')  # 可改 'vgg', 'squeeze'

    results = []
    files = sorted(os.listdir(folder_ref))

    for f in tqdm(files, desc="Processing"):
        ref_path = os.path.join(folder_ref, f)
        test_path = os.path.join(folder_test, f)
        if not os.path.exists(test_path):
            print(f"Warning: {test_path} not found, skip")
            continue

        try:
            psnr_val, ssim_val, lpips_val = calculate_metrics(ref_path, test_path, lpips_fn)
            results.append({
                "filename": f,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "LPIPS": lpips_val
            })
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # 保存结果
    df = pd.DataFrame(results)
    # 追加平均值
    df.loc["Average"] = df.mean(numeric_only=True)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    # 修改为你的文件夹路径
    # folder_ref = "/home/huiluo/Data/FateZero/result/attribute/dog_ddim_5.11/sample/step_0_1_0"
    folder_ref = "/data/huiluo/C-BDIA/result/style/DDIM50/train_samples"
    folder_test = "/data/huiluo/C-BDIA/result/style/jeep_watercolor_260910-160959/sample/step_0_1_0"
    # folder_test = "/home/huiluo/Data/FateZero/result/attribute/dog_ddim_5.11/sample/step_0_1_0"
    evaluate_folder(folder_ref, folder_test, output_csv="metrics_results.csv")
