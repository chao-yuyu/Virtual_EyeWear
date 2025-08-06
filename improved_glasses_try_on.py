import cv2
import numpy as np
import mediapipe as mp
import math

class ImprovedGlassesTryOn:
    def __init__(self):
        """初始化 MediaPipe Face Mesh 模型"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def extract_glasses_region(self, glasses_image):
        """
        從眼鏡圖片中提取實際的眼鏡區域，去除多餘的透明背景
        
        Args:
            glasses_image: 原始眼鏡圖像 (RGBA格式)
            
        Returns:
            tuple: (裁剪後的眼鏡圖像, 邊界框座標)
        """
        print("正在提取眼鏡區域...")
        
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
            print("警告：未找到非透明區域")
            return glasses_image, (0, 0, glasses_image.shape[1], glasses_image.shape[0])
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # 添加一些邊距以確保完整性
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(glasses_image.shape[1] - x, w + 2 * margin)
        h = min(glasses_image.shape[0] - y, h + 2 * margin)
        
        # 裁剪眼鏡區域
        cropped_glasses = glasses_image[y:y+h, x:x+w]
        
        print(f"眼鏡區域提取完成:")
        print(f"  原始尺寸: {glasses_image.shape[1]}x{glasses_image.shape[0]}")
        print(f"  裁剪後尺寸: {w}x{h}")
        print(f"  邊界框: ({x}, {y}, {w}, {h})")
        
        return cropped_glasses, (x, y, w, h)
    
    def detect_face_landmarks(self, image):
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
    
    def get_eye_measurements(self, landmarks):
        """
        精確測量眼部尺寸和位置
        
        Args:
            landmarks: MediaPipe 檢測到的特徵點
            
        Returns:
            dict: 眼部測量數據
        """
        # 關鍵特徵點索引
        left_eye_outer = landmarks[33]      # 左眼外角
        left_eye_inner = landmarks[133]     # 左眼內角
        right_eye_outer = landmarks[362]    # 右眼外角  
        right_eye_inner = landmarks[263]    # 右眼內角
        
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
        
        # 計算眼睛寬度（外角到內角的距離）
        left_eye_width = abs(left_eye_outer[0] - left_eye_inner[0])
        right_eye_width = abs(right_eye_inner[0] - right_eye_outer[0])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        # 計算眼睛高度
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # 計算眼鏡的理想寬度（兩個眼睛寬度 + 鼻樑寬度）
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
        
        measurements = {
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center,
            'left_eye_outer': left_eye_outer,
            'left_eye_inner': left_eye_inner,
            'right_eye_outer': right_eye_outer,
            'right_eye_inner': right_eye_inner,
            'nose_bridge': nose_bridge,
            'eye_center_distance': eye_center_distance,
            'avg_eye_width': avg_eye_width,
            'avg_eye_height': avg_eye_height,
            'ideal_glasses_width': ideal_glasses_width,
            'nose_bridge_width': nose_bridge_width,
            'eye_angle': eye_angle,
            'glasses_center': (glasses_center_x, glasses_center_y)
        }
        
        print("眼部測量結果:")
        print(f"  兩眼中心距離: {eye_center_distance:.1f} pixels")
        print(f"  平均眼睛寬度: {avg_eye_width:.1f} pixels")
        print(f"  平均眼睛高度: {avg_eye_height:.1f} pixels")
        print(f"  鼻樑寬度: {nose_bridge_width:.1f} pixels")
        print(f"  理想眼鏡寬度: {ideal_glasses_width:.1f} pixels")
        print(f"  眼睛角度: {eye_angle:.2f}°")
        print(f"  眼鏡中心: {measurements['glasses_center']}")
        
        return measurements
    
    def calculate_glasses_scale(self, glasses_width, eye_measurements):
        """
        根據眼部測量數據計算眼鏡的最佳縮放比例
        
        Args:
            glasses_width: 眼鏡圖像的實際寬度
            eye_measurements: 眼部測量數據
            
        Returns:
            float: 縮放比例
        """
        # 獲取理想的眼鏡寬度
        target_width = eye_measurements['ideal_glasses_width']
        
        # 計算基本縮放比例
        basic_scale = target_width / glasses_width
        
        # 考慮眼鏡應該稍微大一點以覆蓋整個眼部區域
        # 一般眼鏡會比純眼部測量寬度大 10-20%
        adjustment_factor = 1.15
        final_scale = basic_scale * adjustment_factor
        
        # 限制縮放範圍，避免過大或過小
        final_scale = max(0.1, min(final_scale, 2.0))
        
        print(f"縮放計算:")
        print(f"  眼鏡原始寬度: {glasses_width} pixels")
        print(f"  目標寬度: {target_width:.1f} pixels")
        print(f"  基本縮放比例: {basic_scale:.3f}")
        print(f"  調整後縮放比例: {final_scale:.3f}")
        
        return final_scale
    
    def create_lens_mask(self, glasses_image):
        """
        創建鏡片遮罩，用於實現半透明效果
        
        Args:
            glasses_image: 眼鏡圖像 (RGBA格式)
            
        Returns:
            tuple: (鏡框遮罩, 鏡片遮罩)
        """
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
    
    def transform_glasses(self, glasses_image, scale_factor, rotation_angle):
        """
        變換眼鏡圖像（縮放和旋轉）
        
        Args:
            glasses_image: 眼鏡圖像
            scale_factor: 縮放比例
            rotation_angle: 旋轉角度
            
        Returns:
            tuple: (變換後的眼鏡圖像, 鏡框遮罩, 鏡片遮罩)
        """
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
    
    def blend_glasses(self, face_image, glasses_image, frame_mask, lens_mask, center_pos, lens_opacity=0.4):
        """
        將眼鏡混合到人臉圖像上
        
        Args:
            face_image: 人臉圖像
            glasses_image: 眼鏡圖像
            frame_mask: 鏡框遮罩
            lens_mask: 鏡片遮罩
            center_pos: 眼鏡中心位置
            lens_opacity: 鏡片不透明度 (0-1)
            
        Returns:
            混合後的圖像
        """
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
            print("警告：眼鏡位置超出圖像邊界")
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
        
        # 混合鏡片（半透明）
        blended = blended * (1 - lens_mask_3d * lens_opacity) + glasses_crop * lens_mask_3d * lens_opacity
        
        # 將結果放回原圖
        result[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        
        print(f"眼鏡混合完成:")
        print(f"  位置: ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
        print(f"  實際尺寸: {actual_w}x{actual_h}")
        
        return result
    
    def wear_glasses(self, face_image_path, glasses_image_path, output_path=None, lens_opacity=0.4):
        """
        主函數：將眼鏡佩戴到人臉上
        
        Args:
            face_image_path: 人臉圖像路徑
            glasses_image_path: 眼鏡圖像路徑
            output_path: 輸出路徑（可選）
            lens_opacity: 鏡片不透明度
            
        Returns:
            處理後的圖像，失敗時返回 None
        """
        try:
            print("=== 開始虛擬眼鏡試戴 ===")
            
            # 1. 讀取圖像
            print("\n1. 讀取圖像...")
            face_image = cv2.imread(face_image_path)
            glasses_image = cv2.imread(glasses_image_path, cv2.IMREAD_UNCHANGED)
            
            if face_image is None or glasses_image is None:
                print("錯誤：無法讀取圖像文件")
                return None
            
            print(f"人臉圖像: {face_image.shape}")
            print(f"眼鏡圖像: {glasses_image.shape}")
            
            # 2. 提取眼鏡區域
            print("\n2. 提取眼鏡區域...")
            glasses_cropped, bbox = self.extract_glasses_region(glasses_image)
            
            # 3. 檢測人臉特徵點
            print("\n3. 檢測人臉特徵點...")
            landmarks = self.detect_face_landmarks(face_image)
            if landmarks is None:
                print("錯誤：未檢測到人臉")
                return None
            
            # 4. 測量眼部尺寸
            print("\n4. 測量眼部尺寸...")
            eye_measurements = self.get_eye_measurements(landmarks)
            
            # 5. 計算縮放比例
            print("\n5. 計算眼鏡縮放比例...")
            scale_factor = self.calculate_glasses_scale(
                glasses_cropped.shape[1], 
                eye_measurements
            )
            
            # 6. 變換眼鏡
            print("\n6. 變換眼鏡...")
            transformed_glasses, frame_mask, lens_mask = self.transform_glasses(
                glasses_cropped, 
                scale_factor, 
                eye_measurements['eye_angle']
            )
            
            # 7. 混合圖像
            print("\n7. 混合圖像...")
            result = self.blend_glasses(
                face_image,
                transformed_glasses,
                frame_mask,
                lens_mask,
                eye_measurements['glasses_center'],
                lens_opacity
            )
            
            # 8. 保存結果
            if output_path:
                cv2.imwrite(output_path, result)
                print(f"\n結果已保存到: {output_path}")
            
            print("\n=== 虛擬眼鏡試戴完成 ===")
            return result
            
        except Exception as e:
            print(f"錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主程序"""
    try_on = ImprovedGlassesTryOn()
    
    # 設置文件路徑
    face_path = "240_F_466446411_VYFCWgiDL7LkWsdcRaG3aX8aCfe7jpMu.jpg"
    glasses_path = "pngegg.png"
    output_path = "/home/asri/style_change/improved_result5.jpg"
    
    # 執行眼鏡試戴
    result = try_on.wear_glasses(
        face_image_path=face_path,
        glasses_image_path=glasses_path,
        output_path=output_path,
        lens_opacity=0.4  # 鏡片 40% 不透明度
    )
    
    if result is not None:
        print(f"\n✅ 眼鏡試戴成功完成！")
        print(f"📁 結果已保存到: {output_path}")
        print(f"🖼️  圖像尺寸: {result.shape[1]}x{result.shape[0]}")
        
        # 可選：創建一個調整大小的預覽版本
        preview_path = "/home/asri/style_change/improved_result_preview.jpg"
        h, w = result.shape[:2]
        if h > 600 or w > 600:
            scale = min(600/h, 600/w)
            new_h, new_w = int(h*scale), int(w*scale)
            preview_img = cv2.resize(result, (new_w, new_h))
            cv2.imwrite(preview_path, preview_img)
            print(f"🔍 預覽圖像已保存到: {preview_path}")
        
    else:
        print("❌ 眼鏡試戴失敗")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    print(f"\n程序執行完成，退出代碼: {exit_code}")
    exit(exit_code) 