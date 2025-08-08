#!/usr/bin/env python3
"""
測試基於眉毛對齊的眼鏡試戴功能
"""

import cv2
import os
from glasses_service import glasses_service

def test_eyebrow_alignment():
    """測試眉毛對齊功能"""
    
    # 測試圖像路徑
    test_image_path = "face_with_landmarks.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"測試圖像不存在: {test_image_path}")
        print("請確保有一個名為 'face_with_landmarks.jpg' 的測試圖像")
        return
    
    # 讀取測試圖像
    face_image = cv2.imread(test_image_path)
    if face_image is None:
        print(f"無法讀取圖像: {test_image_path}")
        return
    
    print("正在檢測面部特徵點...")
    
    # 檢測面部特徵點
    landmarks = glasses_service.detect_face_landmarks(face_image)
    if landmarks is None:
        print("未檢測到面部特徵點")
        return
    
    print(f"檢測到 {len(landmarks)} 個特徵點")
    
    # 獲取眼部測量數據
    eye_measurements = glasses_service.get_eye_measurements(landmarks)
    
    print("\n=== 測量結果 ===")
    print(f"眼睛角度: {eye_measurements['eye_angle']:.2f}°")
    print(f"眉毛角度: {eye_measurements['eyebrow_angle']:.2f}°")
    print(f"眼鏡旋轉角度: {eye_measurements['glasses_rotation_angle']:.2f}°")
    print(f"眼間距離: {eye_measurements['eye_center_distance']:.1f} 像素")
    print(f"理想眼鏡寬度: {eye_measurements['ideal_glasses_width']:.1f} 像素")
    
    # 生成調試圖像
    debug_image = glasses_service.draw_debug_landmarks(face_image, landmarks)
    debug_output_path = "debug_landmarks.jpg"
    cv2.imwrite(debug_output_path, debug_image)
    print(f"\n調試圖像已保存到: {debug_output_path}")
    
    # 測試眼鏡試戴（如果有可用的眼鏡）
    available_glasses = glasses_service.get_available_glasses()
    if available_glasses:
        print(f"\n找到 {len(available_glasses)} 副眼鏡，測試第一副...")
        
        first_glasses = available_glasses[0]
        print(f"測試眼鏡: {first_glasses.name} (ID: {first_glasses.id})")
        
        # 執行眼鏡試戴
        result_image = glasses_service.try_on_glasses(
            face_image=face_image,
            glasses_id=first_glasses.id,
            output_path="test_result.jpg"
        )
        
        if result_image is not None:
            print("眼鏡試戴成功！結果已保存到: test_result.jpg")
        else:
            print("眼鏡試戴失敗")
    else:
        print("\n沒有找到可用的眼鏡文件")
    
    print("\n測試完成！")

if __name__ == "__main__":
    test_eyebrow_alignment() 