import time
import os
import argparse
import numpy as np
import cv2
import torch
from torch.backends import cudnn
from torchvision import transforms
from face_detection import RetinaFace
from PIL import Image
from model import SixDRepNet
import utils
import visualization

def parse_args():
    """Parse input arguments
    """
    parser = argparse.ArgumentParser(
        prog="Neo6D",
        description="6DRepNet head pose estimation for video editing",
        epilog="""CSV format: frame_time,x,y,x_rotation,y_rotation,z_rotation,size
        Where size = width * height (of face box).
        """)
    
    parser.add_argument("--gpu", 
        dest="gpu_id", help="GPU device id to use (Default=0)", default=0, type=int
    )
    parser.add_argument("--model",
        dest="model_path", help="Path to model file", default="", type=str, required=True
    )
    parser.add_argument( "--source", 
        dest="source_path", help="Source video's path", default="", type=str, required=True
    )
    parser.add_argument("--save_video",
        dest="save_video", help="Save video with visualization (Default=False)",
        default=False, type=bool
    )
    parser.add_argument("--save_csv",
        dest="save_csv", help="Save csv (Default=True)", default=True, type=bool
    )
    parser.add_argument("--cpu",
        dest="cpu", help="CPU only mode (Default=False)", default=False, type=bool)
    
    args = parser.parse_args()
    return args

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])]
                                     )

def main():
    args = parse_args()
    if not args.cpu and torch.cuda.is_available():
        print('GPU is available')
    else:
        print('GPU is not available. CPU only not implemented.')
        return
    
    cudnn.enabled = True
    gpu_id = args.gpu_id
    video_path = args.source_path
    model_path = args.model_path
    isSaveCSV = args.save_csv
    isSaveVideo = args.save_video
    
    if os.path.exists(model_path):
        print('Model exists')
    else:
        print('File does not exist')
        return
    
    out_csv_path = file_name = os.path.splitext(os.path.basename(video_path))[0] + '.csv'
    if isSaveCSV and os.path.exists(out_csv_path):
        if (input("Overwrite out .csv? y/n : ") == "y"):
            open(out_csv_path, 'w').close()
            
    #Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No file")
        return
    
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)
    
    detector = RetinaFace(gpu_id=gpu_id)
    #Load model to CPU

    saved_state_dict = torch.load(os.path.join(
        model_path), map_location="cpu")

    if "model_state_dict" in saved_state_dict:
        model.load_state_dict(saved_state_dict["model_state_dict"])
    else:
        model.load_state_dict(saved_state_dict)
        
    #Move model to GPU
    model.cuda(gpu_id)

    #Test the model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        
    if isSaveVideo:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Define the codec
        out_file = f"{os.path.splitext(os.path.basename(video_path))[0]}_viz.mp4"
        out = cv2.VideoWriter(os.path.join("", out_file), fourcc, fps, (frame_width, frame_height))

    #Main loop
    with torch.no_grad(): #Reduce memory consumption
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = detector(frame)

            is_face_detected = False #Face isDetected?
            
            for box, landmarks, score in faces:
                if score < 0.95:
                    continue
                
                #Bounding box
                [x_min, y_min, x_max, y_max] = [int(coord) for coord in box[:4]]
                
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                
                margin = int(0.2 * min(bbox_width, bbox_height))    #Calculate box margin
                
                
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = x_max + margin
                y_max = y_max + margin
                
                #Crop face
                img = frame[y_min:y_max, x_min:x_max] #Region of Interest, np slicing
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)
                
                img = torch.Tensor(img[None, :]).cuda(gpu_id)
                
                c = cv2.waitKey(1)
                if c == 27: #ESC
                    break
                                
                #Get rotation
                start = time.time()
                R_pred = model(img)
                end = time.time()
                print(f"Head pose estimation: {(end - start) * 1000:.2f} ms")
                
                #Calculate euler angles
                euler = utils.compute_euler_angles_from_rotation_matrices(
                            R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()
                
                #(tdx, tdy) is the head center
                tdx = x_min + int(.5*(x_max-x_min))
                tdy = y_min + int(.5*(y_max-y_min))
                
                #(x, y, z) rotation in deg
                x_rotation = -p_pred_deg
                y_rotation = y_pred_deg
                z_rotation = r_pred_deg
                
                visualization.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, tdx, tdy)
                cv2.drawMarker(frame, (int(tdx), int(tdy)), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
                frame = visualization.draw_labeled_crosshair(frame, (int(tdx), int(tdy)), f"({tdx}, {tdy})")
                is_face_detected = True 
            
            #Write to file
            frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if is_face_detected and isSaveCSV:
                utils.write_data_to_file(video_path, frame_time, tdx, tdy, 
                               x_rotation, y_rotation, z_rotation, (x_max - x_min) * (y_max - y_min))
            if isSaveVideo:
                out.write(frame)
                
            cv2.imshow("Calculating pose", frame)
    # Release everything when finished
    cap.release()
    if isSaveVideo:
        out.release()
    
    if isSaveCSV:
        utils.append_line_to_top(out_csv_path,
            "frame_time,x,y,x_rotation,y_rotation,z_rotation,size")
    
    print("Success!")
    return
    
            
if __name__ == '__main__':
    main()
    