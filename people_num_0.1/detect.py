import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


def detect( cfg_file = "./detect.cfg"):
    
    with open(cfg_file) as f:
        content = f.readlines()
        for line in content:
            temp = [i for i in line.replace('\t','  ').split("#")[0].strip().split(' ') if i != '']
            if temp == []:
                continue
 
            if "weight" in temp:
                weight = temp[1] #self.ip_pattern.findall(temp)[0][0]
                print("[CONFIG INFO] weight : %s " % weight)
                continue
            elif "cfg_path" in temp:
                cfg_path = temp[1]
                print("[CONFIG INFO] cfg file path : %s" % cfg_path)
                continue
            elif "image_dir" in temp:
                image_dir = temp[1]
                print("[CONFIG INFO] input dir : %s" % image_dir)
                continue
            elif "output_dir" in temp:
                output_dir = temp[1]
                print("[CONFIG INFO] output dir : %s" %  output_dir)
                continue
            elif "img_size" in temp:
                img_size = int(temp[1])
                print("[CONFIG INFO] img size : %d" %  img_size)
                continue
            elif "conf_thres" in temp:
                conf_thres = float(temp[1])
                print("[CONFIG INFO] confidence threshold of object : %f" %  conf_thres)
                continue
            elif "nms_thres" in temp:
                nms_thres = float(temp[1])
                print("[CONFIG INFO] nms threshold : %f" %  nms_thres)
                continue
            elif "save_image" in temp:
                save_image = True if temp[1] == "True" else False
                print("[CONFIG INFO] whether to save image : ", save_image)
                continue
            elif "save_subimage" in temp:
                save_subimage = True if temp[1] == "True" else False
                print("[CONFIG INFO] whether to save detected subimages : ", save_subimage)
                continue
            elif "webcam" in temp:
                webcam = True if temp[1] == "True" else False
                print("[CONFIG INFO] use webcam as input : ", webcam)
                continue
            elif "save_txt" in temp:
                save_txt = True if temp[1] == "True" else False
                print("[CONFIG INFO] save text ? : ", save_txt)
                continue
            elif "class_data" in temp:
                class_data = temp[1]
                print("[CONFIG INFO] path of class data : ", class_data)
                continue




    device = torch_utils.select_device()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # delete output folder
    os.makedirs(output_dir)  # make new output folder

    # Initialize model
    model = Darknet(cfg_path, img_size)

    # Load weights
    if weight.endswith('.pt'):  # pytorch format
        if weight.endswith('yolov3.pt') and not os.path.exists(weight):
            if platform in ('darwin', 'linux'):  # linux/macos
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weight)
        model.load_state_dict(torch.load(weight, map_location = device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weight)

    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_image = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImagesFromDir(image_dir, long_side=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(class_data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()

        basename = os.path.basename(path).split(".")[0]

        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            pass
            #print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        #if ONNX_EXPORT:
        #    torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
        #    return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()
            
            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for index, (*xyxy, conf, cls_conf, cls) in enumerate(detections):
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, cls_conf * conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                if save_subimage: #save_detect:
                    (x1, y1), (x2, y2) = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    subimg = im0[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(output_dir, basename + "_sub_%d.jpg" % index), subimg)

        else:
            continue


        print('Done. (%.1fms)' % (time.time()*1000 - t*1000))

        save_path = os.path.join(output_dir, basename + ".jpg")
        if save_image:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

    #if save_images and platform == 'darwin':  # macos
    #   os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    with torch.no_grad():
        detect( cfg_file = "./cfg.cfg")
