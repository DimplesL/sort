import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from deep_sort_utils.model import Net
from deep_sort_utils.resnet import resnet18


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        # self.net = Net(reid=True)
        self.net = resnet18(11)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        model_dict = self.net.state_dict()
        # 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
        state_dict = {k: v for k, v in state_dict.items() if (k in model_dict and 'fc' not in k)}
        # state_dict = torch.load(model_path, map_location=torch.device(self.device))[
        #     'net_dict']
        self.net.load_state_dict(state_dict, strict=False)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        # self.size = (64, 128)
        self.size = (32, 32)
        # self.norm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        self.net.eval()

        self.norm = transforms.Compose([
            # transforms.Resize((56, 128)),  # h, w
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2471, 0.2435, 0.2616))
        ])

    def predict(self, im_crops):
        im_batch = []
        for img in im_crops:
            img = self.pre_process(img)
            im_batch.append(img)
        tensor = torch.cat([self.norm(im).unsqueeze(0) for im in im_batch], dim=0).float()
        # tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self.net(tensor)
            # probabilities = output.sigmoid().detach().numpy()
        return output.detach().cpu().numpy()
        # return probabilities

    @staticmethod
    def get_rescale_size(src_h: int, src_w: int, target_h: int, target_w: int) -> \
            ((int, int), (int, int, int, int)):
        """
        按长边等比例缩放，短边pad 0
        :param src_h: 源尺寸高
        :param src_w: 源尺寸宽
        :param target_h: 目标尺寸高
        :param target_w: 目标尺寸宽
        :return: （缩放后高，缩放后宽），（左边需要pad的宽度，右边需要pad的宽度，上边需要pad的宽度，下边需要pad的宽度）
        """
        # 等比例缩放
        scale = max(src_h / target_h, src_w / target_w)
        new_h, new_w = int(src_h / scale), int(src_w / scale)
        # padding
        left_more_pad, top_more_pad = 0, 0
        if new_w % 2 != 0:
            left_more_pad = 1
        if new_h % 2 != 0:
            top_more_pad = 1
        left = right = (target_w - new_w) // 2
        top = bottom = (target_h - new_h) // 2
        left += left_more_pad
        top += top_more_pad
        return (new_h, new_w), (left, right, top, bottom)

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """
        对cv2读取的单张BGR图像进行图像等比例伸缩，空余部分pad 0
        :param image: cv2读取的bgr格式图像， (h, w, 3)
        :return: 等比例伸缩后的图像， (h, w, 3)
        """
        h, w = image.shape[:2]
        target_h, target_w = self.size[0], self.size[1]
        (new_h, new_w), (left, right, top, bottom) = self.get_rescale_size(h, w, target_h, target_w)

        # 等比例缩放
        image = cv2.resize(image, (new_w, new_h))
        # padding
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow(f'show', image)
        # cv2.waitKey(0)
        return image

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("/Users/qiuyurui/Desktop/lindau_000000_000019_leftImg8bit.png")[:, :, (2, 1, 0)]
    # extr = Extractor("/Users/qiuyurui/Desktop/models_file/deep_sort_car-ckpt.t7")
    extr = Extractor("/Users/qiuyurui/Projects/PycharmProjects/PyTorch_CIFAR10/state_dicts/resnet18.pt")
    feature = extr.predict(img)
    # feature = extr(np.expand_dims(img, axis=0))
    print(feature.shape)
