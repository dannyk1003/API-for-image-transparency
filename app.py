import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from starlette.staticfiles import StaticFiles
import cv2
import numpy as np
import shutil


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
original_image = None
processed_image = None


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

    # 如果需要，將圖片轉換為 CV_8UC3 格式
    if img_copy.dtype != np.uint8 or img_copy.shape[2] != 3:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGBA2RGB)

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
async def upload(request: Request, image: UploadFile = File(...)):
    global original_image

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    original_image = img

    # 將原始圖片儲存為 PNG 格式
    file_path = "static/original_image.png"
    cv2.imwrite(file_path, img)

    # 將處理後的圖片儲存為 PNG 格式
    processed_img = RMBG(original_image)
    processed_file_path = "static/processed_image.png"
    cv2.imwrite(processed_file_path, processed_img)

    # 讀取 HTML 模板
    with open("index.html", "r", encoding="utf-8") as file:
        template = file.read()

    # 將回應內容填入 HTML 模板
    response_content = """
    <h2>圖片上傳成功！</h2>
    <h3>原始圖片：</h3>
    <img src="{original_image_url}" alt="Original Image" style="max-width: 500px;">
    <h3>照片去背後的圖片：</h3>
    <img src="{processed_image_url}" alt="Processed Image" style="max-width: 500px;">
    """.format(
        original_image_url=f"{request.base_url}/static/original_image.png",
        processed_image_url=f"{request.base_url}/static/processed_image.png",
    )

    html_content = template.format(response_content=response_content)

    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)