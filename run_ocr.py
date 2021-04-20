# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import cv2
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from skimage import io
from PIL import Image, ImageDraw, ImageFont, ImageFont

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from craft import CRAFT
from utils import *

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from collections import OrderedDict

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='models/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.6, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
# parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
# parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_file', default='/data/', type=str, help='path to input image')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

image_list = [args.test_file]

def detect(net, image):
    text_box = get_textbox(net, image, args.text_threshold, args.link_threshold, 
                           args.low_text, args.cuda, args.poly, refine_net)
    horizontal_list, free_list = group_text_box(text_box, (optimal_num_chars is None))
    return horizontal_list, free_list

if __name__ == '__main__':
    print('starting...')
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './models/transformerocr.pth'
    config['device'] = 'cpu'        # set device to use cpu
    config['cnn']['pretrained']=False
    config['predictor']['beamsearch']=False

    # load net
    net = CRAFT()     # initialize CRAFT
    print('Loading regconition weights from checkpoint (' + config['weights'] + ')')
    predictor = Predictor(config)       # initialize VietOCR


    print('Loading detection weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    refine_net = None

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        
        image = imgproc.loadImage(image_path)
        h, w, _ = image.shape
        board = np.zeros((h, w, 3), np.uint8)
        board[:] = (255, 255, 255)

        unicode_font = ImageFont.truetype("./fonts/Montserrat-Bold.ttf", 70)     # load unicode font

        # bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        regs = []
        boxes = []
        optimal_num_chars=None
        print("Detecting...")
        horizontal_list, free_list = detect(net, image)
        print("----------------------------------------")

        for bbox in horizontal_list:
            bbox = [abs(i) for i in bbox]
            reg = predictor.predict(Image.fromarray(image[bbox[2]:bbox[3], bbox[0]:bbox[1], :]))
            regs.append(reg)
            # tl, tr, br, bl
            boxes.append([(bbox[0], bbox[2]), (bbox[1], bbox[2]), (bbox[1], bbox[3]), (bbox[0], bbox[3])])
            # image = cv2.rectangle(image,(bbox[0], bbox[2]), (bbox[1], bbox[3]), (255,0,0), 3)
            # board = Image.fromarray(board)
            # draw = ImageDraw.Draw(board)
            # board = cv2.putText(board, reg, (bbox[0], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,0), thickness=5)
            # draw.text((bbox[0], bbox[3]), reg, font=unicode_font, fill=(0,0,0))
            # board = np.array(board)
            print(reg)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw_img = draw_ocr_box_txt(image, boxes, regs, font_path="./fonts/Montserrat-Bold.ttf")

        draw_img_save = "./inference_results/"
        if not os.path.exists(draw_img_save):
            os.makedirs(draw_img_save)

        cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(args.test_file)),
                draw_img[:, :, ::-1])
        
        # if image.shape[1] < 500:
        #     scale_coef = 1
        # else:
        #     scale_coef = image.shape[1] / 500
        # board = cv2.resize(board,(int(image.shape[1]/scale_coef), int(image.shape[0]/scale_coef)))
        # # image = cv2.resize(image,(int(image.shape[1]/scale_coef), int(image.shape[0]/scale_coef)))
        # image_concate = np.hstack((image,board))
        # image_concate = cv2.resize(image_concate,(int(image_concate.shape[1]/scale_coef), int(image_concate.shape[0]/scale_coef)))
        # cv2.imshow('detected', image_concate)
        # cv2.imshow('asfasf', draw_img)
        # cv2.waitKey(0)