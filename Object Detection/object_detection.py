#Object Detection using Single-Shot Multi box Detector (SSD)

#Importing necessary packages
import torch
from torch.autograd import Variable #convert the tensors in torch variables
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap #preprocessing part
from ssd import build_ssd
import imageio #library to apply the detection (similiar to pil)

#Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1) #from RBG to GRB
    x = Variable(x.unsqueeze(0)) #Getting the batch (it will always be 0), 
    #the Variabel class trasform in a torch variable (with a tensor and gradient)
    y = net(x)
    detections = y.data #getting the values
    scale = torch.Tensor([width, height, width, height])
    #scale represents the normalization, and it uses width and height twice
    # because the first two represent the upper left corner of our detection
    # and the two last one the lower right corner
    
    #detections = [batch, number of classes, number of occurence, (score, x0, y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0 ] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), cv2.LINE_AA)
            j += 1
    
    return frame

#Creating the SSD Neural Network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

#Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

#Doing some object detection on a video
reader = imageio.get_reader("funny_dog.mp4")
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()


