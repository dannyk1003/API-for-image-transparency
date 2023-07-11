from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np
import time
import cvzone




app = FastAPI()
original_image = None

@app.post("/upload/")
async def upload(image: UploadFile = File(...)):
    global original_image

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # 檢查圖片是否具有 alpha 通道(透明度0[最透明]~255[最不透明])，如果沒有則添加
    if img.shape[2] == 3:
        alpha_channel = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
        img = np.dstack((img, alpha_channel))
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
        # 將背景設置為透明(這邊已背景為白色當範例)
        img = original_image
        img[np.all(img == [255, 255, 255, 255], axis=2)] = [0, 0, 0, 0]

        # 將圖片轉換為 PNG 格式
        retval, buffer = cv2.imencode('.png', img)
        transparent_image = buffer.tobytes()

        # img = RMBG(original_image)
        
        return Response(content=transparent_image, media_type='image/png')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)