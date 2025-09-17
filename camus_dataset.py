import os, glob, re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple


class CAMUSMedicalDataset(Dataset):
    """
    CAMUS医学超声数据集加载器
    支持图像+edge+mask+sketch+文本描述的多模态加载
    """
    def __init__(self, data_root: str, size: int = 256, split: str = "train", use_text: bool = True):
        self.data_root = data_root
        self.size = size
        self.use_text = use_text
        
        # 子文件夹路径
        self.img_dir = os.path.join(data_root, "images")
        self.edge_dir = os.path.join(data_root, "edge")
        self.mask_dir = os.path.join(data_root, "masks") 
        self.sketch_dir = os.path.join(data_root, "sketch")
        
        # 收集所有图像文件
        img_patterns = ["*.png", "*.jpg", "*.jpeg"]
        self.img_paths = []
        for pattern in img_patterns:
            self.img_paths.extend(glob.glob(os.path.join(self.img_dir, pattern)))
        
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
            
        # 按文件名排序，确保一致性
        self.img_paths.sort()
        
        # 简单的train/val分割（80/20）
        if split == "train":
            self.img_paths = self.img_paths[:int(0.8 * len(self.img_paths))]
        elif split == "val":
            self.img_paths = self.img_paths[int(0.8 * len(self.img_paths)):]
            
        print(f"[CAMUS {split}] Loaded {len(self.img_paths)} samples")
    
    def __len__(self):
        return len(self.img_paths)
    
    def parse_filename(self, filepath: str) -> Dict[str, str]:
        """
        从文件名解析医学信息
        例：patient0001_2CH_ED.png → {"patient": "0001", "view": "2CH", "phase": "ED", "mirrored": False}
        """
        basename = os.path.splitext(os.path.basename(filepath))[0]
        
        # 处理mirrored前缀
        mirrored = False
        if basename.startswith("mirrored_"):
            mirrored = True
            basename = basename[9:]  # 去掉"mirrored_"
        
        # 正则匹配：patient0001_2CH_ED
        match = re.match(r'patient(\d+)_(\w+)_(\w+)', basename)
        if match:
            patient_id = match.group(1)
            view = match.group(2)  # 2CH, 4CH等
            phase = match.group(3)  # ED, ES等
        else:
            # 如果不匹配，使用默认值
            patient_id = "0000"
            view = "2CH"
            phase = "ED"
        
        return {
            "patient": patient_id,
            "view": view,
            "phase": phase, 
            "mirrored": mirrored,
            "basename": basename
        }
    
    def filename_to_text(self, parsed: Dict[str, str]) -> str:
        """
        将解析的文件名信息转换为医学文本描述
        """
        view_mapping = {
            "2CH": "apical 2-chamber view",
            "4CH": "apical 4-chamber view", 
            "PLAX": "parasternal long-axis view",
            "PSAX": "parasternal short-axis view"
        }
        
        phase_mapping = {
            "ED": "end-diastole",
            "ES": "end-systole"
        }
        
        view_text = view_mapping.get(parsed["view"], f"{parsed['view']} view")
        phase_text = phase_mapping.get(parsed["phase"], parsed["phase"])
        
        # 构建完整描述
        description = f"{view_text}, {phase_text}"
        
        # 可选：添加病人ID信息
        # description = f"patient {parsed['patient']}, {description}"
        
        return description
    
    def load_condition_image(self, img_path: str, cond_dir: str, default_value: float = 0.0) -> np.ndarray:
        """
        根据主图像路径加载对应的条件图像（edge/mask/sketch）
        """
        basename = os.path.basename(img_path)
        cond_path = os.path.join(cond_dir, basename)
        
        if os.path.exists(cond_path):
            cond_img = cv2.imread(cond_path, cv2.IMREAD_GRAYSCALE)
            if cond_img is not None:
                return cond_img.astype(np.float32)
        
        # 如果条件图像不存在，返回填充值
        return np.full((self.size, self.size), default_value, dtype=np.float32)
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        图像预处理：归一化、resize、pad
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # 1-99百分位归一化（对医学图像更稳定）
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
        
        # Resize并保持宽高比
        H, W = img.shape[:2]
        scale = min(self.size / H, self.size / W)
        new_h, new_w = int(H * scale), int(W * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 居中pad到目标尺寸
        pad_h = (self.size - new_h) // 2
        pad_w = (self.size - new_w) // 2
        img_padded = np.pad(img_resized, 
                           ((pad_h, self.size - new_h - pad_h), 
                            (pad_w, self.size - new_w - pad_w)), 
                           mode='constant', constant_values=0)
        
        return img_padded
    
    def __getitem__(self, idx: int) -> Dict:
        img_path = self.img_paths[idx]
        
        # 1. 加载主图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        img = self.preprocess_image(img)
        
        # 2. 加载条件图像
        edge = self.preprocess_image(self.load_condition_image(img_path, self.edge_dir, 0.0))
        mask = self.preprocess_image(self.load_condition_image(img_path, self.mask_dir, 0.0))
        sketch = self.preprocess_image(self.load_condition_image(img_path, self.sketch_dir, 0.5))
        
        # 3. 生成fan条件（合成扇形，模拟超声探头扇形视野）
        fan = self.generate_fan_mask(self.size)
        
        # 4. 解析文件名并生成文本描述
        parsed = self.parse_filename(img_path)
        text_desc = self.filename_to_text(parsed) if self.use_text else None
        
        # 转换为tensor
        # 主图像：[-1,1]范围（适配VAR训练）
        img_tensor = torch.from_numpy(img)[None].float() * 2 - 1  # [1,H,W]
        
        # 条件图像：[0,1]范围
        edge_tensor = torch.from_numpy(edge)[None].float()   # [1,H,W]
        mask_tensor = torch.from_numpy(mask)[None].float()   # [1,H,W]
        sketch_tensor = torch.from_numpy(sketch)[None].float() # [1,H,W]
        fan_tensor = torch.from_numpy(fan)[None].float()     # [1,H,W]
        
        return {
            "img": img_tensor,                    # 主图像 [-1,1]
            "edge": edge_tensor,                  # 边缘 [0,1]
            "mask": mask_tensor,                  # 掩码 [0,1]  
            "sketch": sketch_tensor,              # 草图 [0,1]
            "fan": fan_tensor,                    # 扇形 [0,1]
            "text_desc": text_desc,               # 文本描述
            "patient_id": parsed["patient"],      # 病人ID
            "view": parsed["view"],               # 视图类型
            "phase": parsed["phase"],             # 心脏期相
            "path": img_path,                     # 原始路径
        }
    
    def generate_fan_mask(self, size: int) -> np.ndarray:
        """
        生成扇形掩码，模拟超声探头的扇形视野
        """
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, size),
            np.linspace(-1, 1, size),
            indexing='ij'
        )
        
        # 极坐标
        rr = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        
        # 扇形参数：半径<0.9，角度在[-60°, 60°]范围内
        angle_min, angle_max = -np.pi/3, np.pi/3
        fan = ((rr < 0.9) & (theta > angle_min) & (theta < angle_max)).astype(np.float32)
        
        return fan


# 测试函数
def test_camus_dataset():
    """测试CAMUS数据集加载器"""
    dataset = CAMUSMedicalDataset(
        data_root="/home/tao/pxt/VAR/CAMUS_Gen",
        size=256,
        split="train",
        use_text=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试加载几个样本
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {sample['img'].shape}")
        print(f"  Text: '{sample['text_desc']}'")
        print(f"  Patient: {sample['patient_id']}")
        print(f"  View: {sample['view']}, Phase: {sample['phase']}")
        print(f"  Edge shape: {sample['edge'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        

if __name__ == "__main__":
    test_camus_dataset()