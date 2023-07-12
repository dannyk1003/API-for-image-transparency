from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np


app = FastAPI()
original_image = None

def RMBG(img):
    img_copy = np.copy(img)
    # 轉BGR
    bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGBA2BGR)

    # 轉HSV顏色空間
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 假設背景在左上區域
    x1, y1 = 0, 0
    x2, y2 = 10, 10
    background_roi = hsv[y1:y2, x1:x2]
    # 提取色調 H 通道
    hue_channel = background_roi[:, :, 0]
    # 計算背景區 H 最大值與最小值
    h_min = np.min(hue_channel)
    h_max = np.max(hue_channel)

    # 定義背景顏色範圍
    lower_bound = np.array([h_min-5, 100, 100])  # 最小的HSV值
    upper_bound = np.array([h_max+5, 255, 255])  # 最大的HSV值

    # 創建範圍遮罩
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 遮罩應用，背景變白
    img_copy[mask == 255] = 255

    # 使用openCV的GrabCut法找出邊緣
    mask2 = np.zeros(img_copy.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, img_copy.shape[1] - 1, img_copy.shape[0] - 1)

    cv2.grabCut(img_copy, mask2, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask3 = np.where((mask2 == 2) | (mask2 == 0), 0, 1).astype('uint8')

    img_copy = img_copy*mask3[:,:,np.newaxis]

    #背景變透明
    if img_copy.shape[2] == 3:
        alpha_channel = np.full((img_copy.shape[0], img_copy.shape[1]), 255, dtype=np.uint8)
        img_copy = np.dstack((img_copy, alpha_channel))

    img_copy[np.all(img_copy == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

    return img_copy

@app.post("/upload/")
async def upload(image: UploadFile = File(...)):
    global original_image

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    original_image = img
    
    # 儲存原始圖片，轉 PNG 格式
    retval, buffer = cv2.imencode('.png', img)
    img = buffer.tobytes()

    return Response(content=img, media_type='image/png')

@app.get("/remove_background/")
async def remove_background():
    global original_image

    if original_image is None:
        return {"message": "upload image first!!"}
    else:
        img = RMBG(original_image)

        # 將圖片轉換為 PNG 格式
        retval, buffer = cv2.imencode('.png', img)
        transparent_image = buffer.tobytes()
        
        return Response(content=transparent_image, media_type='image/png')


if __name__ == "__main__":
    #連結上fastapi，預設port為8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 程式運作流程
# 匯入必要的模組和套件，包括FastAPI、File、UploadFile等。
# 創建FastAPI應用程式的實例。
# 定義一個全局變數original_image，用於存儲上傳的原始圖片。
# 定義一個RMBG函數，用於移除圖片的背景。
# 定義一個/upload/的POST路由，用於處理圖片上傳的請求。在該路由中，從上傳的圖片中讀取內容並轉換為OpenCV可處理的格式。將轉換後的圖片存儲到original_image變數中，然後將原始圖片以PNG格式返回給客戶端。
# 定義一個/remove_background/的GET路由，用於處理移除背景的請求。在該路由中，檢查original_image變數是否為空，如果為空則返回錯誤訊息。如果不為空，則將原始圖片傳遞給RMBG函數進行背景移除操作，並將結果以PNG格式返回給客戶端。
# 執行FastAPI應用程式，使用uvicorn伺服器運行在本地主機上的端口8000。