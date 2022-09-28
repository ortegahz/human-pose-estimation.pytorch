from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import logging
import numpy as np

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths

from core.inference import get_final_preds
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

from utils.transforms import get_affine_transform

num_joints = 17  # coco
image_width = 192
image_height = 256
aspect_ratio = image_width * 1.0 / image_height
pixel_std = 200


def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)


def _xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def main():
    # {"bbox": [81.5511842832181, 233.66049769970982, 121.17168271594619, 351.7913157042058],
    # "category_id": 1, "image_id": 17905, "score": 0.9997398257255554}
    image_file = '/media/manu/samsung/pics/000000017905.jpg'
    box = [81.5511842832181, 233.66049769970982, 121.17168271594619, 351.7913157042058]
    score = 0.9997398257255554

    th_pt = 0.5

    colours = np.random.rand(num_joints, 3) * 255

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=False
    )

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    kpt_db = []
    center, scale = _box2cs(box)
    joints_3d = np.zeros((num_joints, 3), dtype=np.float)
    joints_3d_vis = np.ones(
        (num_joints, 3), dtype=np.float)
    kpt_db.append({
        'image': image_file,
        'center': center,
        'scale': scale,
        'score': score,
        'joints_3d': joints_3d,
        'joints_3d_vis': joints_3d_vis,
    })

    logger = logging.getLogger(__name__)

    data_numpy = cv2.imread(
        image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if data_numpy is None:
        logger.error('=> fail to read {}'.format(image_file))
        raise ValueError('Fail to read {}'.format(image_file))

    db_rec = kpt_db[0]
    joints = db_rec['joints_3d']
    joints_vis = db_rec['joints_3d_vis']

    c = db_rec['center']
    s = db_rec['scale']
    score = db_rec['score'] if 'score' in db_rec else 1
    r = 0

    trans = get_affine_transform(c, s, r, [image_width, image_height])
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(image_width), int(image_height)),
        flags=cv2.INTER_LINEAR)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    input = transform(input)
    input = input.unsqueeze(0)

    model.eval()
    output = model(input)
    # output_save = output.detach().cpu().numpy()
    # np.savetxt(os.path.join('/home/manu/tmp', 'output_save_infer.txt'), output_save.flatten(), fmt="%f", delimiter="\n")
    # pred, _ = get_max_preds(output)

    c = np.expand_dims(c, 0)
    s = np.expand_dims(s, 0)
    preds, maxvals = get_final_preds(
        config, output.detach().clone().cpu().numpy(), c, s)

    cv2.namedWindow(image_file, cv2.WINDOW_NORMAL)

    for i, (pred, maxval) in enumerate(zip(preds[0], maxvals[0])):
        color = colours[i, :]
        x = pred[0]
        y = pred[1]
        if maxval[0] > th_pt:
            cv2.circle(data_numpy, (int(x), int(y)), 3, color, -1)
            cv2.putText(data_numpy, f'{maxval[0]:.2f}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow(image_file, data_numpy)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
