import numpy as np
import torch, os, cv2
from torchvision import transforms

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET, U2NETP

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

class U2NETPredictor:
    def __init__(self, model_name="u2net_human_seg"):
        self.device = "cuda"
        #self.net = U2NETP(3,1).to(self.device)
        self.net = U2NET(3,1).to(self.device)
        model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
        self.net.load_state_dict(torch.load(model_dir, map_location=self.device))

        # Evaluation mode
        self.net.eval()

        self.transform = transforms.Compose([RescaleT(240),
                                                ToTensorLab(flag=0)])

    def predict(self, image):
        with torch.no_grad():
            h, w = image.shape[0:2]
            label = np.zeros((h,w))
            label = np.expand_dims(label, axis=2)
            sample = {"imidx":np.array([0]), "image": image, "label": label}

            sample = self.transform(sample)

            inputs_test = sample["image"]
            inputs_test = inputs_test.type(torch.FloatTensor)
            inputs_test = inputs_test.unsqueeze(axis=0).to(self.device)

            d1 = self.net(inputs_test)[0]

            # normalization
            predict = d1[:,0,:,:]
            predict = normPRED(predict)

            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()

            #im = Image.fromarray(predict_np*255).convert('RGB')
            im = predict_np*255
            im = cv2.resize(im, (w,h))
            pred = cv2.dilate(im,np.ones((3,3)),iterations = 3)
            pred = cv2.erode(pred,np.ones((3,3)),iterations = 3)
            
            return pred


if False:
    img1 = cv2.imread(r"U-2-Net\test_data\test_human_images\5-mental-skills-of-successful-athletes-image.jpg")
    img2 = cv2.imread(r"U-2-Net\test_data\test_human_images\2019-LADIES-NIGHT-2ND-GOMES.jpg")
    u = U2NETPredictor()

    import cv2
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    h,w = frame.shape[:2]
    bg = cv2.imread(r"U-2-Net\test_data\test_human_images\5-mental-skills-of-successful-athletes-image.jpg")
    bg = cv2.resize(bg, (w,h))
    imgs = []
    index = 0
    while(True):
        ret, frame = vid.read()
        if index%3 ==0:
            pred =  u.predict(frame)
            pred = cv2.dilate(pred,np.ones((3,3)),iterations = 3)
            pred = cv2.erode(pred,np.ones((3,3)),iterations = 3)

        coords = np.where(pred < 150)
        frame[coords] = bg[coords]
        
        #bg[pred == 255] = frame[pred == 255]
        #cv2.imwrite("frame.jpg", frame)
        #cv2.imwrite("pred.jpg", pred)
        cv2.imshow('frame', frame)
        cv2.imshow('out', pred)
        
        index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    vid.release()
    cv2.destroyAllWindows()
