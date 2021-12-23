import cv2
import numpy as np
import os.path as osp


class BackgroundIterator(object):
    def __init__(self, image_size, para):
        self.image_shape = image_size
        self.imsize = image_size[0]
        assert isinstance(para, str)
        para = para.split('-')
        self.type = para[0]
        self.idx = para[1]
        if self.type == 'Background':
            self.background = self._load_background('multiworld/core/background/')
        elif self.type == 'Video':
            self.background = self._load_video('multiworld/core/background/video')
            self.current_frame = 0
        else:
            raise NotImplementedError

    def _load_background(self, path):
        # color order follows OpenCV style - (B, G, R)
        if int(self.idx) == 0:
            # default = cyan (1, 1, 0.6)
            return None
        elif int(self.idx) == 1:
            # pink (1, 0, 1)
            img = np.ones([self.imsize, self.imsize, 3], dtype=np.int32) * 255
            img[:, :, 1] = 0
            return img
        elif int(self.idx) == 2:
            # yellow (0, 1, 1)
            img = np.ones([self.imsize, self.imsize, 3], dtype=np.int32) * 255
            img[:, :, 0] = 0
            return img
        elif int(self.idx) == 3:
            img = cv2.imread(path + 'man1.png', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 4:
            img = cv2.imread(path + 'man2.png', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 5:
            img = cv2.imread(path + 'robot1.png', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 6:
            img = cv2.imread(path + 'robot2.png', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 7:
            img = cv2.imread(path + 'robotarm1.png', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 8:
            img = cv2.imread(path + 'robotarm2.png', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 9:
            img = cv2.imread(path + 'factory.png', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 10:
            img = cv2.imread(path + 'lightening.png', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 11:
            img = cv2.imread(path + 'robot_spot.jpg', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 12:
            img = cv2.imread(path + 'factory_car.jpg', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 13:
            img = cv2.imread(path + 'mountain_scenery.jpg', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 14:
            img = cv2.imread(path + 'sea_scenery.jpg', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 15:
            img = cv2.imread(path + 'wildcat.jpg', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 16:
            img = cv2.imread(path + 'panda_stack.jpg', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 17:
            img = cv2.imread(path + 'robots_manufacturing_cropped.jpg', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 18:
            img = cv2.imread(path + 'single_robot_manufacturing.jpg', 1)
            return cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
        elif int(self.idx) == 19:
            # light blue
            img = np.ones([self.imsize, self.imsize, 3], dtype=np.int32) * 255
            img[:, :, 0] = 255
            img[:, :, 1] = 179
            img[:, :, 2] = 153
            return img
        elif int(self.idx) == 20:
            # light green
            img = np.ones([self.imsize, self.imsize, 3], dtype=np.int32) * 255
            img[:, :, 0] = 147
            img[:, :, 1] = 196
            img[:, :, 2] = 125
            return img
        elif int(self.idx) == 21:
            # grey
            img = np.ones([self.imsize, self.imsize, 3], dtype=np.int32) * 255
            img[:, :, 0] = 204
            img[:, :, 1] = 204
            img[:, :, 2] = 204
            return img
        elif int(self.idx) == 22:
            # blue
            img = np.ones([self.imsize, self.imsize, 3], dtype=np.int32) * 255
            img[:, :, 0] = 250
            img[:, :, 1] = 100
            img[:, :, 2] = 0
            return img
        elif int(self.idx) == 23:
            # red
            img = np.ones([self.imsize, self.imsize, 3], dtype=np.int32) * 255
            img[:, :, 0] = 51
            img[:, :, 1] = 0
            img[:, :, 2] = 255
            return img
        elif int(self.idx) == 24:
            # purple
            img = np.ones([self.imsize, self.imsize, 3], dtype=np.int32) * 255
            img[:, :, 0] = 255
            img[:, :, 1] = 0
            img[:, :, 2] = 204
            return img
        else:
            raise NotImplementedError

    def _load_video(self, path):
        # change this index range if we have more video
        assert 0 <= int(self.idx) <= 15, "video index out of range"
        video_path = osp.abspath(osp.join(path, 'video' + str(self.idx) + '.mp4'))
        cap = cv2.VideoCapture(video_path)
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
                       dtype=np.uint8)
        i, ret = 0, True
        while (i < n and ret):
            ret, frame = cap.read()
            # we convert color channels in `render` function
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return buf

    def reset(self):
        if self.type == 'Video':
            self.current_frame = 0

    def render(self):
        if self.background is None:
            return None
        elif self.type == 'Background':
            return self.background.copy()[::-1, :, ::-1]
        elif self.type == 'Video':
            img = self.background[self.current_frame % len(self.background)].copy()
            img = cv2.resize(img, self.image_shape, interpolation=cv2.INTER_CUBIC)
            self.current_frame += 1
            return img[::-1, :, ::-1]