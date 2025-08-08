"""
API 測試腳本
"""

import requests
import json

def test_api():
    """測試 API 功能"""
    base_url = "http://localhost:8000"
    
    print("🧪 開始測試 API...")
    
    # 1. 測試健康檢查
    print("\n1. 測試健康檢查...")
    response = requests.get(f"{base_url}/health")
    print(f"狀態碼: {response.status_code}")
    print(f"響應: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    
    # 2. 測試獲取眼鏡列表
    print("\n2. 測試獲取眼鏡列表...")
    response = requests.get(f"{base_url}/glasses")
    print(f"狀態碼: {response.status_code}")
    glasses_list = response.json()
    print(f"眼鏡數量: {len(glasses_list)}")
    for glasses in glasses_list:
        print(f"  - {glasses['id']}: {glasses['name']} ({glasses['type']})")
    
    # 3. 測試根據類型獲取眼鏡
    print("\n3. 測試獲取墨鏡列表...")
    response = requests.get(f"{base_url}/glasses/type/sunglasses")
    print(f"狀態碼: {response.status_code}")
    sunglasses = response.json()
    print(f"墨鏡數量: {len(sunglasses)}")
    
    print("\n4. 測試獲取一般眼鏡列表...")
    response = requests.get(f"{base_url}/glasses/type/regular")
    print(f"狀態碼: {response.status_code}")
    regular_glasses = response.json()
    print(f"一般眼鏡數量: {len(regular_glasses)}")
    
    # 5. 測試獲取特定眼鏡信息
    print("\n5. 測試獲取特定眼鏡信息...")
    response = requests.get(f"{base_url}/glasses/glasses_1")
    print(f"狀態碼: {response.status_code}")
    print(f"響應: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

def test_try_on_api(image_path: str, glasses_id: str):
    """測試眼鏡試戴 API"""
    base_url = "http://localhost:8000"
    
    print(f"\n🧪 測試眼鏡試戴 API...")
    print(f"圖像路徑: {image_path}")
    print(f"眼鏡ID: {glasses_id}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'glasses_id': glasses_id}
            
            response = requests.post(f"{base_url}/try-on", files=files, data=data)
            
        print(f"狀態碼: {response.status_code}")
        result = response.json()
        print(f"響應: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get('success') and result.get('result_url'):
            print(f"✅ 試戴成功！結果URL: {base_url}{result['result_url']}")
        else:
            print(f"❌ 試戴失敗: {result.get('message')}")
            
    except FileNotFoundError:
        print(f"❌ 圖像文件不存在: {image_path}")
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")

if __name__ == "__main__":
    # 基本 API 測試
    test_api()
    
    # 眼鏡試戴測試（如果有測試圖像）
    test_image_path = "images.jpeg"  # 替換為實際的測試圖像路徑
    test_glasses_id = "glasses_1"    # 測試眼鏡ID
    
    import os
    if os.path.exists(test_image_path):
        test_try_on_api(test_image_path, test_glasses_id)
    else:
        print(f"\n⚠️  測試圖像不存在: {test_image_path}")
        print("請將測試圖像放在當前目錄或修改 test_image_path 變量") 