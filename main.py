from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import tempfile
import os
import uuid
import shutil
from datetime import datetime

from glasses_config import GlassesType, glasses_manager
from glasses_service import glasses_service

# 創建 FastAPI 應用
app = FastAPI(
    title="虛擬眼鏡試戴 API",
    description="使用 MediaPipe 和 OpenCV 實現的虛擬眼鏡試戴服務",
    version="1.0.0"
)

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 創建必要的目錄
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
STATIC_DIR = "static"
GLASSES_DIR = "glasses"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(GLASSES_DIR, exist_ok=True)

# 掛載靜態文件服務
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images/glasses", StaticFiles(directory=GLASSES_DIR), name="glasses")

# Pydantic 模型
class GlassesResponse(BaseModel):
    """眼鏡信息響應模型"""
    id: str
    name: str
    type: str
    description: str
    lens_opacity: float
    image_url: str  # 添加圖片URL字段

class TryOnResponse(BaseModel):
    """試戴響應模型"""
    success: bool
    message: str
    result_url: Optional[str] = None
    glasses_info: Optional[GlassesResponse] = None

class ErrorResponse(BaseModel):
    """錯誤響應模型"""
    success: bool
    message: str
    error_code: Optional[str] = None

@app.get("/")
async def root():
    """根路徑 - 重定向到 Web 界面"""
    return FileResponse('static/index.html')

@app.get("/glasses", response_model=List[GlassesResponse])
async def get_all_glasses():
    """獲取所有可用的眼鏡"""
    try:
        glasses_list = glasses_service.get_available_glasses()
        return [
            GlassesResponse(
                id=glasses.id,
                name=glasses.name,
                type=glasses.type.value,
                description=glasses.description,
                lens_opacity=glasses.lens_opacity,
                image_url=f"/images/glasses/{glasses.filename}"
            )
            for glasses in glasses_list
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取眼鏡列表失敗: {str(e)}")

@app.get("/glasses/{glasses_id}", response_model=GlassesResponse)
async def get_glasses_by_id(glasses_id: str):
    """根據ID獲取眼鏡信息"""
    try:
        glasses_info = glasses_manager.get_glasses_by_id(glasses_id)
        return GlassesResponse(
            id=glasses_info.id,
            name=glasses_info.name,
            type=glasses_info.type.value,
            description=glasses_info.description,
            lens_opacity=glasses_info.lens_opacity,
            image_url=f"/images/glasses/{glasses_info.filename}"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取眼鏡信息失敗: {str(e)}")

@app.get("/glasses/type/{glasses_type}", response_model=List[GlassesResponse])
async def get_glasses_by_type(glasses_type: str):
    """根據類型獲取眼鏡列表"""
    try:
        if glasses_type not in ["regular", "sunglasses"]:
            raise HTTPException(status_code=400, detail="無效的眼鏡類型，必須是 'regular' 或 'sunglasses'")
        
        glasses_type_enum = GlassesType(glasses_type)
        glasses_list = glasses_manager.get_glasses_by_type(glasses_type_enum)
        
        return [
            GlassesResponse(
                id=glasses.id,
                name=glasses.name,
                type=glasses.type.value,
                description=glasses.description,
                lens_opacity=glasses.lens_opacity,
                image_url=f"/images/glasses/{glasses.filename}"
            )
            for glasses in glasses_list
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取眼鏡列表失敗: {str(e)}")

@app.post("/try-on", response_model=TryOnResponse)
async def try_on_glasses(
    image: UploadFile = File(..., description="人臉圖像文件"),
    glasses_id: str = Form(..., description="眼鏡ID")
):
    """虛擬眼鏡試戴"""
    
    # 驗證文件類型
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="上傳的文件必須是圖像格式")
    
    # 驗證眼鏡ID
    try:
        glasses_info = glasses_manager.get_glasses_by_id(glasses_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"未找到眼鏡 ID: {glasses_id}")
    
    # 生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    # 保存上傳的圖像
    upload_filename = f"{timestamp}_{unique_id}_input.jpg"
    upload_path = os.path.join(UPLOAD_DIR, upload_filename)
    
    try:
        # 讀取上傳的圖像
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_image is None:
            raise HTTPException(status_code=400, detail="無法解析上傳的圖像")
        
        # 保存輸入圖像（用於調試）
        cv2.imwrite(upload_path, face_image)
        
        # 生成結果文件名
        result_filename = f"{timestamp}_{unique_id}_{glasses_id}_result.jpg"
        result_path = os.path.join(RESULT_DIR, result_filename)
        
        # 執行眼鏡試戴
        result_image = glasses_service.try_on_glasses(
            face_image=face_image,
            glasses_id=glasses_id,
            output_path=result_path
        )
        
        if result_image is None:
            return TryOnResponse(
                success=False,
                message="眼鏡試戴失敗，可能是因為未檢測到人臉或其他處理錯誤"
            )
        
        # 構建結果URL
        result_url = f"/results/{result_filename}"
        
        return TryOnResponse(
            success=True,
            message="眼鏡試戴成功",
            result_url=result_url,
            glasses_info=GlassesResponse(
                id=glasses_info.id,
                name=glasses_info.name,
                type=glasses_info.type.value,
                description=glasses_info.description,
                lens_opacity=glasses_info.lens_opacity,
                image_url=f"/images/glasses/{glasses_info.filename}"
            )
        )
        
    except Exception as e:
        return TryOnResponse(
            success=False,
            message=f"處理過程中發生錯誤: {str(e)}"
        )

@app.post("/debug-landmarks")
async def debug_landmarks(image: UploadFile = File(..., description="人臉圖像文件")):
    """調試端點：顯示面部特徵點檢測結果"""
    
    # 驗證文件類型
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="上傳的文件必須是圖像格式")
    
    # 生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    try:
        # 讀取上傳的圖像
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_image is None:
            raise HTTPException(status_code=400, detail="無法解析上傳的圖像")
        
        # 檢測面部特徵點
        landmarks = glasses_service.detect_face_landmarks(face_image)
        if landmarks is None:
            raise HTTPException(status_code=400, detail="未檢測到人臉特徵點")
        
        # 繪製調試特徵點
        debug_image = glasses_service.draw_debug_landmarks(face_image, landmarks)
        
        # 保存調試圖像
        debug_filename = f"{timestamp}_{unique_id}_debug.jpg"
        debug_path = os.path.join(RESULT_DIR, debug_filename)
        cv2.imwrite(debug_path, debug_image)
        
        # 獲取眼部測量數據
        eye_measurements = glasses_service.get_eye_measurements(landmarks)
        
        return {
            "success": True,
            "message": "特徵點檢測成功",
            "debug_image_url": f"/results/{debug_filename}",
            "measurements": {
                "eye_angle": eye_measurements["eye_angle"],
                "eyebrow_angle": eye_measurements["eyebrow_angle"],
                "glasses_rotation_angle": eye_measurements["glasses_rotation_angle"],
                "eye_center_distance": eye_measurements["eye_center_distance"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理失敗: {str(e)}")

@app.get("/results/{filename}")
async def get_result_image(filename: str):
    """獲取處理結果圖像"""
    file_path = os.path.join(RESULT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="結果圖像不存在")
    
    return FileResponse(
        path=file_path,
        media_type='image/jpeg',
        filename=filename
    )

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    try:
        # 檢查眼鏡文件是否存在
        validation_results = glasses_manager.validate_glasses_files()
        missing_files = [k for k, v in validation_results.items() if not v]
        
        if missing_files:
            return {
                "status": "warning",
                "message": f"部分眼鏡文件缺失: {missing_files}",
                "glasses_validation": validation_results
            }
        
        return {
            "status": "healthy",
            "message": "服務運行正常",
            "glasses_count": len(glasses_manager.get_all_glasses()),
            "glasses_validation": validation_results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"健康檢查失敗: {str(e)}"
        }

@app.delete("/cleanup")
async def cleanup_files():
    """清理臨時文件"""
    try:
        # 清理上傳文件（保留最近24小時的）
        cleanup_count = 0
        current_time = datetime.now().timestamp()
        
        for directory in [UPLOAD_DIR, RESULT_DIR]:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        file_time = os.path.getctime(file_path)
                        # 刪除超過24小時的文件
                        if current_time - file_time > 24 * 3600:
                            os.remove(file_path)
                            cleanup_count += 1
        
        return {
            "success": True,
            "message": f"清理完成，刪除了 {cleanup_count} 個過期文件"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"清理失敗: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 啟動虛擬眼鏡試戴 API 服務...")
    print("📖 API 文檔: http://localhost:8000/docs")
    print("🔍 健康檢查: http://localhost:8000/health")
    print("👓 眼鏡列表: http://localhost:8000/glasses")
    
    uvicorn.run(
        "main:app",  # 使用 import string 而不是直接傳遞 app
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 