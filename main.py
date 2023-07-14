from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/upload', response_class=HTMLResponse)
async def upload(request: Request, image: UploadFile = UploadFile(...)):
    contents = await image.read()


    origin_path = 'templates/origin.PNG'
    with open(origin_path, "wb") as file:
        file.write(contents)

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    img = RMBG(img)
    retval, buffer = cv2.imencode('.png', img)
    img = buffer.tobytes()

    process_path = 'templates/process.PNG'
    with open(process_path, "wb") as file:
        file.write(img)

    return templates.TemplateResponse('my_html.html', {"request": request, "origin_url": "/origin", 'process_url': '/process'})

@app.get('/origin')
async def origin():
    file_path = 'templates/origin.PNG'
    return FileResponse(file_path)


@app.get('/process')
async def process():
    file_path = 'templates/process.PNG'
    return FileResponse(file_path)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5555)