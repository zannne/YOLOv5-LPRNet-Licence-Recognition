import argparse
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *

def detect():
    # Initialize
    device = torch.device("cuda:0")
    out = './licence/demo/rec_result'
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder

    # Load yolov5 model
    model = attempt_load('./licence/weights/yolov5_best.pt', map_location=device)  # load FP32 model
    model = model.cuda()
    print("load det pretrained model successful!")
    imgsz = 640

    dataset = LoadImages('./licence/demo/images/', img_size=imgsz)


    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        # Apply NMS
        conf_thres = 0.4
        iou_thres = 0.5
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for _, det in enumerate(pred):  # detections per image
            p, im0 = path, im0s
            save_path = str(Path(out) / Path(p).name)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for x in det:
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    cv2.rectangle(im0, c1, c2, [118, 187, 254], thickness=3, lineType=cv2.LINE_AA)

            # Save results (image with detections)
            cv2.imwrite(save_path, im0)


if __name__ == '__main__':
    with torch.no_grad():
        detect()
