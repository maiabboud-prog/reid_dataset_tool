import cv2
import numpy as np
import onnxruntime as ort
import albumentations as A
from scipy.spatial import distance
from albumentations.pytorch import ToTensorV2
import time

class FeatureExtraction:
    def __init__(
        self,
        # onnx_path="./models/osnet/osnet_ain_x1_0_C.onnx",
        # onnx_path="./models/osnet/osnet_ain_x1_0_D.onnx",
        # onnx_path="./models/osnet/osnet_ain_x1_0_M.onnx",
        onnx_path="../models/osnet_ain_x1_0_MS.onnx",
        # onnx_path="./models/osnet/osnet_x1_0.onnx",

        # onnx_path="./pretrained_models/osnet_x1_0.onnx",
        device="cpu",
    ):
        self.onnx_path = onnx_path
        self.device = device
        if self.device.lower() == "cpu":
            if ort.get_device() == "CUDA":
                print("CUDA available, if you want to switch your device into CUDA")
            self.ort_session = ort.InferenceSession(
                self.onnx_path, providers=["CPUExecutionProvider"]
            )
        elif self.device.lower() == "cuda":
            self.ort_session = ort.InferenceSession(
                self.onnx_path,
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            )
        else:
            raise ValueError("Choose between CPU or CUDA!")
        self.model_height, self.model_width = self.ort_session.get_inputs()[0].shape[
            2:4
        ]
        self.image_augmentation = A.Compose(
            [A.Resize(self.model_height, self.model_width), A.Normalize(), ToTensorV2()]
        )

    def predict_img(self, img):
        image = self._preprocessing_img(img)
        input_onnx = self.ort_session.get_inputs()[0].name
        output_onnx = self.ort_session.run(None, {input_onnx: image})
        return output_onnx

    def _preprocessing_img(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.image_augmentation(image=np.array(image))["image"]
        image = np.expand_dims(image, axis=0)
        return image

    def extract_features_from_person_boxes(self, frame, person_boxes):
        extracted_feats = []
        for x1, y1, x2, y2 in person_boxes:
            crop = frame[y1:y2, x1:x2]
            feat = self.predict_img(crop)[0]
            if feat.ndim == 1:
                feat = np.expand_dims(feat, axis=0)
            extracted_feats.append(feat)
        return extracted_feats

# if __name__ == "__main__":
#     feature_extraction = FeatureExtraction()
#     cam0 = cv2.imread("./sample_image/img0.jpg")
#     cam1 = cv2.imread("./sample_image/img1.jpg")
#     cam3 = cv2.imread("./sample_image/img2.png")

#     t0 = time.time()
#     output0 = feature_extraction.predict_img(cam0)[0]
#     t1 = time.time()

#     output1 = feature_extraction.predict_img(cam1)[0]
#     print("Inference time: %.2f ms" % ((t1 - t0) * 1000))
#     output3 = feature_extraction.predict_img(cam3)[0]
    
#     output0  = output0.squeeze()
#     output1  = output1.squeeze()
#     output3  = output3.squeeze()
#     print(1-distance.cosine(output0, output3))
