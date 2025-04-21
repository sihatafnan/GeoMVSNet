# -*- coding: utf-8 -*-
# @Description: Main process of network testing.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os, time, sys, gc, cv2, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets.data_io import *
from datasets.dtu import DTUDataset
from datasets.tnt import TNTDataset

from models.geomvsnet import GeoMVSNet
from models.utils import *
from models.utils.opts import get_opts


cudnn.benchmark = True

args = get_opts()

def perturb_ref_camera(sample):
    # print("Camera before perturbation (ref and src extrinsics):")
    extrinsics = sample["proj_matrices"]["stage4"][0, :, 0]  # All extrinsics in this batch
    # print(extrinsics)

    for i in range(extrinsics.shape[0]):
        extrinsic = extrinsics[i]
        extrinsic[0, 3] += 5
        extrinsic[1, 3] -= 10
        extrinsic[2, 3] += 12

        d_roll = torch.tensor(1, dtype=torch.float32, device=extrinsic.device)
        d_pitch = torch.tensor(-2, dtype=torch.float32, device=extrinsic.device)
        d_yaw = torch.tensor(1.5, dtype=torch.float32, device=extrinsic.device)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(d_roll), -torch.sin(d_roll)],
            [0, torch.sin(d_roll), torch.cos(d_roll)]
        ], dtype=torch.float32, device=extrinsic.device)

        Ry = torch.tensor([
            [torch.cos(d_pitch), 0, torch.sin(d_pitch)],
            [0, 1, 0],
            [-torch.sin(d_pitch), 0, torch.cos(d_pitch)]
        ], dtype=torch.float32, device=extrinsic.device)

        Rz = torch.tensor([
            [torch.cos(d_yaw), -torch.sin(d_yaw), 0],
            [torch.sin(d_yaw), torch.cos(d_yaw), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=extrinsic.device)

        R_delta = Rz @ Ry @ Rx
        extrinsic[:3, :3] = R_delta @ extrinsic[:3, :3]
        sample_cuda["proj_matrices"]["stage4"][0, i, 0] = extrinsic

    # print("Camera after perturbation (ref and src extrinsics):")
    # print(sample_cuda["proj_matrices"]["stage4"][0, :, 0])

    return sample

def test():
    total_time = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            if args.attack:
                sample = perturb_ref_camera(sample)
            sample_cuda = tocuda(sample)
            start_time = time.time()

            # @Note GeoMVSNet main
            outputs = model(
                sample_cuda["imgs"], 
                sample_cuda["proj_matrices"], sample_cuda["intrinsics_matrices"], 
                sample_cuda["depth_values"], 
                sample["filename"]
            )

            end_time = time.time()
            total_time += end_time - start_time
            outputs = tensor2numpy(outputs)
            del sample_cuda

            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(args.levels)].numpy()
            imgs = sample["imgs"]
            logger.info('Iter {}/{}, Time:{:.3f} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))


            for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"]):
                img = img[0].numpy()    # ref view
                cam = cam[0]            # ref cam

                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                if args.which_dataset == 'dtu':
                    os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                
                # save depth maps
                save_pfm(depth_filename, depth_est)

                # save confidence maps
                confidence_list = [outputs['stage{}'.format(i)]['photometric_confidence'].squeeze(0) for i in range(1,5)]
                photometric_confidence = confidence_list[-1]
                if not args.save_conf_all_stages:
                    save_pfm(confidence_filename, photometric_confidence) 
                else:
                    for stage_idx, photometric_confidence in enumerate(confidence_list):
                        if stage_idx != args.levels - 1:
                            confidence_filename = os.path.join(args.outdir, filename.format('confidence', "_stage"+str(stage_idx)+'.pfm'))
                        else:
                            confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                        save_pfm(confidence_filename, photometric_confidence) 

                # save cams, img
                if args.which_dataset == 'dtu':
                    write_cam(cam_filename, cam)
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)

    torch.cuda.empty_cache()
    gc.collect()
    return total_time, len(TestImgLoader)


def initLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))

    if args.which_dataset == 'tnt':
        logfile = os.path.join(args.logdir, 'TNT-test-' + curTime + '.log')
    else:
        logfile = os.path.join(args.logdir, 'test-' + curTime + '.log')
    
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    if not args.nolog:
        fileHandler = logging.FileHandler(logfile, mode='a')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger initialized.")
    logger.info("Writing logs to file: {}".format(logfile))
    logger.info("Current time: {}".format(curTime))

    settings_str = "All settings:\n"
    for k,v in vars(args).items(): 
        settings_str += '{0}: {1}\n'.format(k,v)
    logger.info(settings_str)

    return logger


if __name__ == '__main__':
    logger = initLogger()

    # dataset, dataloader
    if args.which_dataset == 'dtu':
        test_dataset = DTUDataset(args.testpath, args.testlist, "test", args.n_views, max_wh=(1600, 1200))
    elif args.which_dataset == 'tnt':
        test_dataset = TNTDataset(args.testpath, args.testlist, split=args.split, n_views=args.n_views, img_wh=(-1, 1024), cam_mode=args.cam_mode, img_mode=args.img_mode)

    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # @Note GeoMVSNet model
    model = GeoMVSNet(
        levels=args.levels, 
        hypo_plane_num_stages=[int(n) for n in args.hypo_plane_num_stages.split(",")], 
        depth_interal_ratio_stages=[float(ir) for ir in args.depth_interal_ratio_stages.split(",")],
        feat_base_channel=args.feat_base_channel, 
        reg_base_channel=args.reg_base_channel,
        group_cor_dim_stages=[int(n) for n in args.group_cor_dim_stages.split(",")],
    )
    
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=False)

    model.cuda()
    model.eval()

    test()
