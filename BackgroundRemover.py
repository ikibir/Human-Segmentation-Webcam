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

