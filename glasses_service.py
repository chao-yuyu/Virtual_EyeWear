import cv2
import numpy as np
import mediapipe as mp
import math
import tempfile
import os
from typing import Optional, Tuple, Union
from glasses_config import GlassesManager, GlassesInfo, glasses_manager

class GlassesTryOnService:
    """眼鏡試戴服務類"""
    
    def __init__(self):
        """初始化 MediaPipe Face Mesh 模型和眼鏡管理器"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.glasses_manager = glasses_manager
    
    def extract_glasses_region(self, glasses_image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """從眼鏡圖片中提取實際的眼鏡區域，去除多餘的透明背景"""
        # 獲取 alpha 通道
        if glasses_image.shape[2] == 4:
            alpha = glasses_image[:, :, 3]
        else:
            # 如果沒有 alpha 通道，基於顏色創建遮罩
            gray = cv2.cvtColor(glasses_image, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            glasses_image = np.dstack([glasses_image, alpha])
        
        # 找到非透明區域的邊界框
        coords = cv2.findNonZero(alpha)
        if coords is None:
            return glasses_image, (0, 0, glasses_image.shape[1], glasses_image.shape[0])
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # 添加邊距
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(glasses_image.shape[1] - x, w + 2 * margin)
        h = min(glasses_image.shape[0] - y, h + 2 * margin)
        
        # 裁剪眼鏡區域
        cropped_glasses = glasses_image[y:y+h, x:x+w]
        
        return cropped_glasses, (x, y, w, h)
    
    def detect_face_landmarks(self, image: np.ndarray) -> Optional[list]:
        """使用 MediaPipe 檢測臉部特徵點"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            return landmarks
        return None
    
    def get_eye_measurements(self, landmarks: list) -> dict:
        """精確測量眼部尺寸和位置"""
        # 關鍵特徵點索引
        left_eye_outer = landmarks[33]
        left_eye_inner = landmarks[133]
        right_eye_outer = landmarks[362]
        right_eye_inner = landmarks[263]
        
        # 眼睛上下邊界點
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        
        # 鼻樑點
        nose_bridge = landmarks[6]
        
        # 計算眼睛中心
        left_eye_center = (
            (left_eye_outer[0] + left_eye_inner[0]) // 2,
            (left_eye_top[1] + left_eye_bottom[1]) // 2
        )
        right_eye_center = (
            (right_eye_outer[0] + right_eye_inner[0]) // 2,
            (right_eye_top[1] + right_eye_bottom[1]) // 2
        )
        
        # 計算各種測量值
        eye_center_distance = math.sqrt(
            (right_eye_center[0] - left_eye_center[0])**2 + 
            (right_eye_center[1] - left_eye_center[1])**2
        )
        
        # 計算眼睛寬度
        left_eye_width = abs(left_eye_outer[0] - left_eye_inner[0])
        right_eye_width = abs(right_eye_inner[0] - right_eye_outer[0])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        # 計算眼睛高度
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # 計算眼鏡的理想寬度
        nose_bridge_width = abs(left_eye_inner[0] - right_eye_inner[0])
        ideal_glasses_width = left_eye_width + right_eye_width + nose_bridge_width
        
        # 計算眼睛的角度
        eye_angle = math.degrees(math.atan2(
            right_eye_center[1] - left_eye_center[1],
            right_eye_center[0] - left_eye_center[0]
        ))
        
        # 計算眼鏡中心位置
        glasses_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        glasses_center_y = (left_eye_center[1] + right_eye_center[1]) // 2
        
        return {
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center,
            'eye_center_distance': eye_center_distance,
            'avg_eye_width': avg_eye_width,
            'avg_eye_height': avg_eye_height,
            'ideal_glasses_width': ideal_glasses_width,
            'nose_bridge_width': nose_bridge_width,
            'eye_angle': eye_angle,
            'glasses_center': (glasses_center_x, glasses_center_y)
        }
    
    def calculate_glasses_scale(self, glasses_width: int, eye_measurements: dict) -> float:
        """根據眼部測量數據計算眼鏡的最佳縮放比例"""
        target_width = eye_measurements['ideal_glasses_width']
        basic_scale = target_width / glasses_width
        adjustment_factor = 1.15
        final_scale = basic_scale * adjustment_factor
        
        # 限制縮放範圍
        final_scale = max(0.1, min(final_scale, 2.0))
        
        return final_scale
    
    def create_lens_mask(self, glasses_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """創建鏡片遮罩，用於實現半透明效果"""
        alpha = glasses_image[:, :, 3]
        
        # 使用形態學操作區分鏡框和鏡片
        kernel = np.ones((5, 5), np.uint8)
        
        # 腐蝕操作找到內部區域（鏡片）
        eroded = cv2.erode(alpha, kernel, iterations=2)
        
        # 鏡框遮罩：原始 alpha 減去腐蝕結果
        frame_mask = cv2.subtract(alpha, eroded)
        
        # 鏡片遮罩：腐蝕結果
        lens_mask = eroded
        
        # 清理鏡片遮罩，只保留較大的連通區域
        contours, _ = cv2.findContours(lens_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lens_mask_clean = np.zeros_like(lens_mask)
        
        # 設置最小面積閾值
        min_area = lens_mask.shape[0] * lens_mask.shape[1] * 0.02
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.fillPoly(lens_mask_clean, [contour], 255)
        
        return frame_mask, lens_mask_clean
    
    def transform_glasses(self, glasses_image: np.ndarray, scale_factor: float, rotation_angle: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """變換眼鏡圖像（縮放和旋轉）"""
        h, w = glasses_image.shape[:2]
        
        # 縮放
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        scaled_glasses = cv2.resize(glasses_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 創建遮罩
        frame_mask, lens_mask = self.create_lens_mask(scaled_glasses)
        
        # 如果旋轉角度很小，跳過旋轉
        if abs(rotation_angle) < 2.0:
            return scaled_glasses, frame_mask, lens_mask
        
        # 旋轉
        center = (new_w // 2, new_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # 計算旋轉後的邊界
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w_rot = int((new_h * sin_val) + (new_w * cos_val))
        new_h_rot = int((new_h * cos_val) + (new_w * sin_val))
        
        # 調整旋轉中心
        rotation_matrix[0, 2] += (new_w_rot / 2) - center[0]
        rotation_matrix[1, 2] += (new_h_rot / 2) - center[1]
        
        # 應用旋轉
        rotated_glasses = cv2.warpAffine(scaled_glasses, rotation_matrix, (new_w_rot, new_h_rot))
        rotated_frame_mask = cv2.warpAffine(frame_mask, rotation_matrix, (new_w_rot, new_h_rot))
        rotated_lens_mask = cv2.warpAffine(lens_mask, rotation_matrix, (new_w_rot, new_h_rot))
        
        return rotated_glasses, rotated_frame_mask, rotated_lens_mask
    
    def blend_glasses(self, face_image: np.ndarray, glasses_image: np.ndarray, 
                     frame_mask: np.ndarray, lens_mask: np.ndarray, 
                     center_pos: Tuple[int, int], lens_opacity: float = 0.4) -> np.ndarray:
        """將眼鏡混合到人臉圖像上"""
        result = face_image.copy()
        h_face, w_face = face_image.shape[:2]
        h_glasses, w_glasses = glasses_image.shape[:2]
        
        # 計算眼鏡位置
        start_x = center_pos[0] - w_glasses // 2
        start_y = center_pos[1] - h_glasses // 2
        
        # 邊界檢查和調整
        start_x = max(0, min(start_x, w_face - w_glasses))
        start_y = max(0, min(start_y, h_face - h_glasses))
        
        end_x = min(start_x + w_glasses, w_face)
        end_y = min(start_y + h_glasses, h_face)
        
        # 計算實際可用區域
        actual_w = end_x - start_x
        actual_h = end_y - start_y
        
        if actual_w <= 0 or actual_h <= 0:
            return result
        
        # 裁剪眼鏡和遮罩
        glasses_crop = glasses_image[:actual_h, :actual_w, :3]
        frame_mask_crop = frame_mask[:actual_h, :actual_w]
        lens_mask_crop = lens_mask[:actual_h, :actual_w]
        
        # 獲取人臉區域
        face_region = result[start_y:end_y, start_x:end_x].astype(np.float32)
        glasses_crop = glasses_crop.astype(np.float32)
        
        # 正規化遮罩
        frame_mask_norm = frame_mask_crop.astype(np.float32) / 255.0
        lens_mask_norm = lens_mask_crop.astype(np.float32) / 255.0
        
        # 創建三通道遮罩
        frame_mask_3d = np.stack([frame_mask_norm] * 3, axis=2)
        lens_mask_3d = np.stack([lens_mask_norm] * 3, axis=2)
        
        # 混合鏡框（完全不透明）
        blended = face_region * (1 - frame_mask_3d) + glasses_crop * frame_mask_3d
        
        # 混合鏡片（根據透明度設置）
        blended = blended * (1 - lens_mask_3d * lens_opacity) + glasses_crop * lens_mask_3d * lens_opacity
        
        # 將結果放回原圖
        result[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        
        return result
    
    def try_on_glasses(self, face_image: Union[str, np.ndarray], 
                      glasses_id: str, 
                      output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        主要的眼鏡試戴功能
        
        Args:
            face_image: 人臉圖像路徑或numpy數組
            glasses_id: 眼鏡ID
            output_path: 輸出路徑（可選）
            
        Returns:
            處理後的圖像，失敗時返回 None
        """
        try:
            # 1. 讀取人臉圖像
            if isinstance(face_image, str):
                face_img = cv2.imread(face_image)
                if face_img is None:
                    raise ValueError(f"無法讀取人臉圖像: {face_image}")
            else:
                face_img = face_image
            
            # 2. 獲取眼鏡信息和圖像
            glasses_info = self.glasses_manager.get_glasses_by_id(glasses_id)
            glasses_path = self.glasses_manager.get_glasses_path(glasses_id)
            
            glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
            if glasses_img is None:
                raise ValueError(f"無法讀取眼鏡圖像: {glasses_path}")
            
            # 3. 提取眼鏡區域
            glasses_cropped, _ = self.extract_glasses_region(glasses_img)
            
            # 4. 檢測人臉特徵點
            landmarks = self.detect_face_landmarks(face_img)
            if landmarks is None:
                raise ValueError("未檢測到人臉特徵點")
            
            # 5. 測量眼部尺寸
            eye_measurements = self.get_eye_measurements(landmarks)
            
            # 6. 計算縮放比例
            scale_factor = self.calculate_glasses_scale(
                glasses_cropped.shape[1], 
                eye_measurements
            )
            
            # 7. 變換眼鏡
            transformed_glasses, frame_mask, lens_mask = self.transform_glasses(
                glasses_cropped, 
                scale_factor, 
                eye_measurements['eye_angle']
            )
            
            # 8. 混合圖像（使用眼鏡配置中的透明度設置）
            result = self.blend_glasses(
                face_img,
                transformed_glasses,
                frame_mask,
                lens_mask,
                eye_measurements['glasses_center'],
                glasses_info.lens_opacity
            )
            
            # 9. 保存結果
            if output_path:
                cv2.imwrite(output_path, result)
            
            return result
            
        except Exception as e:
            print(f"眼鏡試戴失敗: {str(e)}")
            return None
    
    def get_available_glasses(self) -> list:
        """獲取所有可用的眼鏡列表"""
        return self.glasses_manager.get_all_glasses()

# 創建全局服務實例
glasses_service = GlassesTryOnService() 