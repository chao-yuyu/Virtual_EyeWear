#!/usr/bin/env python3
"""
簡單的眼鏡試戴測試腳本
"""

import requests
import json

def test_try_on_api():
    """測試眼鏡試戴 API"""
    url = "http://localhost:8000/try-on"
    
    # 測試圖像文件路徑
    image_path = "/home/asri/style_change/pl/240_F_188636224_N3vXpsWNsAjYkT9g1WmQP9ximvByAMVN.jpg"
    glasses_id = "glasses_1"  # 測試墨鏡
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'glasses_id': glasses_id}
            
            print(f"🧪 測試眼鏡試戴 API...")
            print(f"📷 圖像: {image_path}")
            print(f"👓 眼鏡: {glasses_id}")
            print("⏳ 處理中...")
            
            response = requests.post(url, files=files, data=data)
            
        print(f"\n📊 響應狀態: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功: {result.get('success')}")
            print(f"💬 訊息: {result.get('message')}")
            
            if result.get('success') and result.get('result_url'):
                print(f"🖼️  結果URL: http://localhost:8000{result['result_url']}")
                print(f"👓 眼鏡信息: {result['glasses_info']['name']}")
            
            print(f"\n完整響應:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"❌ 請求失敗: {response.text}")
            
    except FileNotFoundError:
        print(f"❌ 圖像文件不存在: {image_path}")
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")

if __name__ == "__main__":
    test_try_on_api() 