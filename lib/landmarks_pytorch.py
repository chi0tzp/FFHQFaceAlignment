"""
Calculate euler angles yaw pitch roll using deep network HopeNet
https://github.com/natanielruiz/deep-head-pose

The face detector used is SFD (taken from face-alignment FAN) https://github.com/1adrianb/face-alignment

"""
from enum import Enum
from torch.utils.model_zoo import load_url
from .sfd.sfd_detector import SFDDetector as FaceDetector
from .fan_model.models import FAN, ResNetDepth
from .fan_model.utils import *


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Initialise the dimensions of the image to be resized and grab the image size
    (h, w) = image.shape[:2]

    # If both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # Check to see if the width is None
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
        scale = r

    # Otherwise, the height is None
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
        scale = r

    if width is not None and height is not None:
        dim = (width, height)

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized, scale


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.
    _2D - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    _2halfD - this points represent the projection of the 3D points into 3D
    _3D - detect the points ``(x,y,z)``` in a 3D space
    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


models_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center and the scale is provided the function will
    return the points also in the original coordinate frame.

    Arguments:
        hm (torch.tensor) -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})

    """
    max, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx = idx + 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if (0 < pX < 63) and (0 < pY < 63):
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig


class LandmarksEstimation:
    def __init__(self, type='3D'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load all needed models - Face detector and Pose detector
        network_size = NetworkSize.LARGE
        network_size = int(network_size)
        if type == '3D':
            self.landmarks_type = LandmarksType._3D
        else:
            self.landmarks_type = LandmarksType._2D
        self.flip_input = False

        # SFD face detection
        path_to_detector = os.path.join(sys.path[0], 'lib/sfd/s3fd-619a316812.pth')
        self.face_detector = FaceDetector(device='cuda', verbose=False, path_to_detector=path_to_detector)

        self.transformations_image = transforms.Compose([transforms.Resize(224),
                                                         transforms.CenterCrop(224), transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])])
        self.transformations = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

        # Initialise the face alignment networks
        self.face_alignment_net = FAN(network_size)
        if self.landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(network_size)
        else:
            network_name = '3DFAN-' + str(network_size)

        fan_weights = load_url(models_urls[network_name], map_location=lambda storage, loc: storage)
        self.face_alignment_net.load_state_dict(fan_weights)

        self.face_alignment_net.to(self.device)
        self.face_alignment_net.eval()

        # Initialise the depth prediction network
        if self.landmarks_type == LandmarksType._3D:
            self.depth_prediction_net = ResNetDepth()

            depth_weights = load_url(models_urls['depth'], map_location=lambda storage, loc: storage)
            depth_dict = {k.replace('module.', ''): v for k, v in depth_weights['state_dict'].items()}
            self.depth_prediction_net.load_state_dict(depth_dict)

            self.depth_prediction_net.to(self.device)
            self.depth_prediction_net.eval()

    def find_landmarks(self, face, image):
        center = torch.FloatTensor(
            [(face[2] + face[0]) / 2.0,
             (face[3] + face[1]) / 2.0])

        center[1] = center[1] - (face[3] - face[1]) * 0.12
        scale = (face[2] - face[0] + face[3] - face[1]) / self.face_detector.reference_scale

        inp = crop_torch(image.unsqueeze(0), center, scale).float().cuda()
        inp = inp.div(255.0)
        out = self.face_alignment_net(inp)[-1]

        if self.flip_input:
            out = out + flip(self.face_alignment_net(flip(inp))
                             [-1], is_label=True)  # patched inp_batch undefined variable error
        out = out.cpu()

        pts, pts_img = get_preds_fromhm(out, center, scale)
        out = out.cuda()

        # Added 3D landmark support
        if self.landmarks_type == LandmarksType._3D:
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            heatmaps = torch.zeros((68, 256, 256), dtype=torch.float32)
            for i in range(68):
                if pts[i, 0] > 0:
                    heatmaps[i] = draw_gaussian(
                        heatmaps[i], pts[i], 2)

            heatmaps = heatmaps.unsqueeze(0)
            heatmaps = heatmaps.to(self.device)
            if inp.shape[2] != heatmaps.shape[2] or inp.shape[3] != heatmaps.shape[3]:
                print(inp.shape)
                print(heatmaps.shape)

            depth_pred = self.depth_prediciton_net(torch.cat((inp, heatmaps), 1)).view(68, 1)
            pts_img = pts_img.cuda()
            pts_img = torch.cat((pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

        else:
            pts, pts_img = pts.view(-1, 68, 2) * 4, pts_img.view(-1, 68, 2)

        return pts_img, out

    def face_detection(self, image):
        image_tensor = torch.tensor(np.transpose(image, (2, 0, 1))).float().cuda()
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0).cuda()
            detected_faces, error, error_index = self.face_detector.detect_from_batch(image_tensor)
        else:
            detected_faces, error, error_index = self.face_detector.detect_from_batch(image_tensor)

        if len(detected_faces[0]) == 0:
            return image

        for face in detected_faces[0]:
            conf = face[4]

            if conf > 0.9:
                x1 = face[0]
                y1 = face[1]
                x2 = face[2]
                y2 = face[3]
                w = x2 - x1
                h = y2 - y1

                cx = int(x1 + w / 2)
                cy = int(y1 + h / 2)

                if h > w:
                    w = h
                    x1_hat = cx - int(w / 2)
                    if x1_hat < 0:
                        x1_hat = 0
                    x2_hat = x1_hat + w

                else:
                    h = w
                    y1_hat = cy - int(h / 2)
                    if y1_hat < 0:
                        y1_hat = 0
                    y2_hat = y1_hat + h

                w_hat = int(w * 1.6)
                h_hat = int(h * 1.6)
                x1_hat = cx - int(w_hat / 2)
                if x1_hat < 0:
                    x1_hat = 0
                y1_hat = cy - int(h_hat / 2)
                if y1_hat < 0:
                    y1_hat = 0
                x2_hat = x1_hat + w_hat
                y2_hat = y1_hat + h_hat
                crop = image.copy()

                crop = crop[y1_hat:y2_hat, x1_hat:x2_hat]

                # print(w_hat, h_hat)
                crop, scale = image_resize(crop, 256, 256)

        return crop

    @torch.no_grad()
    def detect_landmarks(self, image, detected_faces=None):
        if detected_faces is None:
            if len(image.shape) == 3:
                image = image.unsqueeze(0).cuda()
                detected_faces, error, error_index = self.face_detector.detect_from_batch(image)
            else:
                detected_faces, error, error_index = self.face_detector.detect_from_batch(image)

        batch = 0
        num_faces = 0
        em_max = -1
        index_face = 0
        for face in detected_faces[0]:
            conf = face[4]
            w = face[2] - face[0]
            h = face[3] - face[1]
            em = w * h
            if em > em_max:
                em_max = em
                index_face = num_faces
            num_faces += 1

        if self.landmarks_type == LandmarksType._3D:
            landmarks = torch.empty((1, 68, 3), requires_grad=True).cuda()
        else:
            landmarks = torch.empty((1, 68, 2), requires_grad=True).cuda()

        counter = 0
        for face in detected_faces[0]:
            conf = face[4]
            if conf > 0.99 and counter == index_face:
                pts_img, heatmaps = self.find_landmarks(face, image[0])
                landmarks[batch] = pts_img.cuda()
                batch += 1
            counter += 1

        return landmarks, detected_faces
