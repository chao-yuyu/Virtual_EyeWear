from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import os

class GlassesType(Enum):
    """眼鏡類型枚舉"""
    REGULAR = "regular"  # 一般眼鏡（需要透明效果）
    SUNGLASSES = "sunglasses"  # 墨鏡（不需要透明效果）

@dataclass
class GlassesInfo:
    """眼鏡信息類"""
    id: str
    name: str
    filename: str
    type: GlassesType
    lens_opacity: float  # 鏡片透明度（0-1，0為完全透明，1為完全不透明）
    description: str = ""

# 眼鏡配置字典
GLASSES_CONFIG: Dict[str, GlassesInfo] = {
    "glasses_1": GlassesInfo(
        id="glasses_1",
        name="經典黑框墨鏡",
        filename="glasses_1.png",
        type=GlassesType.SUNGLASSES,
        lens_opacity=0.8,  # 墨鏡不透明度較高
        description="時尚黑框墨鏡，適合戶外使用"
    ),
    "glasses_2": GlassesInfo(
        id="glasses_2",
        name="圓框透明眼鏡",
        filename="glasses_2.png",
        type=GlassesType.REGULAR,
        lens_opacity=0.3,  # 一般眼鏡透明度較高
        description="復古圓框設計，適合日常配戴"
    ),
    "glasses_3": GlassesInfo(
        id="glasses_3",
        name="方框透明眼鏡",
        filename="glasses_3.png",
        type=GlassesType.REGULAR,
        lens_opacity=0.3,
        description="現代方框設計，專業商務風格"
    ),
    "glasses_4": GlassesInfo(
        id="glasses_4",
        name="貓眼框透明眼鏡",
        filename="glasses_4.png",
        type=GlassesType.REGULAR,
        lens_opacity=0.3,
        description="優雅貓眼框設計，展現女性魅力"
    ),
    "glasses_5": GlassesInfo(
        id="glasses_5",
        name="運動框透明眼鏡",
        filename="glasses_5.png",
        type=GlassesType.REGULAR,
        lens_opacity=0.3,
        description="運動風格設計，輕量舒適"
    ),
    "glasses_6": GlassesInfo(
        id="glasses_6",
        name="飛行員墨鏡",
        filename="glasses_6.png",
        type=GlassesType.SUNGLASSES,
        lens_opacity=0.8,
        description="經典飛行員款式，永不過時"
    ),
    "glasses_7": GlassesInfo(
        id="glasses_7",
        name="橢圓框透明眼鏡",
        filename="glasses_7.png",
        type=GlassesType.REGULAR,
        lens_opacity=0.3,
        description="溫和橢圓框設計，適合各種臉型"
    ),
    "glasses_8": GlassesInfo(
        id="glasses_8",
        name="大框透明眼鏡",
        filename="glasses_8.png",
        type=GlassesType.REGULAR,
        lens_opacity=0.3,
        description="時尚大框設計，突顯個性"
    ),
    "glasses_9": GlassesInfo(
        id="glasses_9",
        name="粗框透明眼鏡",
        filename="glasses_9.png",
        type=GlassesType.REGULAR,
        lens_opacity=0.3,
        description="簡約粗框設計，低調優雅"
    ),
}

class GlassesManager:
    """眼鏡管理器"""
    
    def __init__(self, glasses_dir: str = "glasses"):
        self.glasses_dir = glasses_dir
        
    def get_all_glasses(self) -> List[GlassesInfo]:
        """獲取所有眼鏡列表"""
        return list(GLASSES_CONFIG.values())
    
    def get_glasses_by_id(self, glasses_id: str) -> GlassesInfo:
        """根據ID獲取眼鏡信息"""
        if glasses_id not in GLASSES_CONFIG:
            raise ValueError(f"未找到眼鏡 ID: {glasses_id}")
        return GLASSES_CONFIG[glasses_id]
    
    def get_glasses_by_type(self, glasses_type: GlassesType) -> List[GlassesInfo]:
        """根據類型獲取眼鏡列表"""
        return [info for info in GLASSES_CONFIG.values() if info.type == glasses_type]
    
    def get_glasses_path(self, glasses_id: str) -> str:
        """獲取眼鏡文件的完整路徑"""
        glasses_info = self.get_glasses_by_id(glasses_id)
        return os.path.join(self.glasses_dir, glasses_info.filename)
    
    def validate_glasses_files(self) -> Dict[str, bool]:
        """驗證所有眼鏡文件是否存在"""
        results = {}
        for glasses_id, info in GLASSES_CONFIG.items():
            file_path = self.get_glasses_path(glasses_id)
            results[glasses_id] = os.path.exists(file_path)
        return results
    
    def add_glasses(self, glasses_info: GlassesInfo) -> None:
        """添加新的眼鏡配置"""
        GLASSES_CONFIG[glasses_info.id] = glasses_info
    
    def remove_glasses(self, glasses_id: str) -> None:
        """移除眼鏡配置"""
        if glasses_id in GLASSES_CONFIG:
            del GLASSES_CONFIG[glasses_id]
        else:
            raise ValueError(f"未找到眼鏡 ID: {glasses_id}")

# 創建全局眼鏡管理器實例
glasses_manager = GlassesManager() 