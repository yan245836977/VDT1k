import os
import numpy as np
import shutil
from cal_iou_prec import *
import warnings
# 忽略所有DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
if __name__ == '__main__':
    trakcers = os.listdir("./VDT1k")

    wstr = "Methods, PR(RGBT), NPR(RGBT), SR(RGBT), PR(RGBD), NPR(RGBD), SR(RGBD)\n"
    for tracker in trakcers:
        # rgbd
        pd_rt = f"./VDT1k/{tracker}/rgbd/"
        gt_rt = "F:/VDT_dataset/annotations/"

        seqs = [i.replace("\n","") for i in open(r"F:\VDT_dataset\test_list.txt").readlines()]

        rgb_succ_score_all1, rgb_prec_score_all1, rgb_norm_prec_score_all1 = [], [], []
        for seq in seqs:
            pd_dir = pd_rt+seq+".txt"
            gt_dir = gt_rt+seq+".txt"



            try:
                pd_bxs = np.loadtxt(pd_dir, delimiter="\t", dtype=np.int16)
            except:
                try:
                    pd_bxs = np.loadtxt(pd_dir, delimiter=' ', dtype=np.int16)
                except:
                    pd_bxs = np.loadtxt(pd_dir, delimiter=',', dtype=np.int16)



            gt_bxs = np.loadtxt(gt_dir,delimiter="\t",dtype=np.int16)

            assert len(pd_bxs)==len(gt_bxs)
            protocol = 1
            # test RGB results
            rgb_succ_score, rgb_prec_score, rgb_norm_prec_score = calc_rgbps_seq_performace(pd_bxs,
                                                                                            gt_bxs,
                                                                                            protocol)

            rgb_succ_score_all1.append(rgb_succ_score)
            rgb_prec_score_all1.append(rgb_prec_score)
            rgb_norm_prec_score_all1.append(rgb_norm_prec_score)

        rgb_succ_score1 = torch.tensor(rgb_succ_score_all1).mean().tolist() * 100
        rgb_prec_score1 = torch.tensor(rgb_prec_score_all1).mean().tolist() * 100
        rgb_norm_prec_score1 = torch.tensor(rgb_norm_prec_score_all1).mean().tolist() * 100

        # rgbd
        pd_rt = f"./VDT1k/{tracker}/rgbt/"
        gt_rt = "F:/VDT_dataset/annotations/"

        seqs = [i.replace("\n","") for i in open(r"F:\VDT_dataset\test_list.txt").readlines()]

        rgb_succ_score_all2, rgb_prec_score_all2, rgb_norm_prec_score_all2 = [], [], []
        for seq in seqs:
            pd_dir = pd_rt+seq+".txt"
            gt_dir = gt_rt+seq+".txt"



            try:
                pd_bxs = np.loadtxt(pd_dir, delimiter="\t", dtype=np.int16)
            except:
                try:
                    pd_bxs = np.loadtxt(pd_dir, delimiter=' ', dtype=np.int16)
                except:
                    pd_bxs = np.loadtxt(pd_dir, delimiter=',', dtype=np.int16)



            gt_bxs = np.loadtxt(gt_dir,delimiter="\t",dtype=np.int16)

            assert len(pd_bxs)==len(gt_bxs)
            protocol = 1
            # test RGB results
            rgb_succ_score, rgb_prec_score, rgb_norm_prec_score = calc_rgbps_seq_performace(pd_bxs,
                                                                                            gt_bxs,
                                                                                            protocol)

            rgb_succ_score_all2.append(rgb_succ_score)
            rgb_prec_score_all2.append(rgb_prec_score)
            rgb_norm_prec_score_all2.append(rgb_norm_prec_score)

        rgb_succ_score2 = torch.tensor(rgb_succ_score_all2).mean().tolist() * 100
        rgb_prec_score2 = torch.tensor(rgb_prec_score_all2).mean().tolist() * 100
        rgb_norm_prec_score2 = torch.tensor(rgb_norm_prec_score_all2).mean().tolist() * 100

        print(f'{tracker} | PR: {rgb_prec_score1:.1f} | NPR: {rgb_norm_prec_score1:.1f}| SR: {rgb_succ_score1:.1f}  ||'
              f'PR: {rgb_prec_score2:.1f} | NPR: {rgb_norm_prec_score2:.1f}| SR: {rgb_succ_score2:.1f}  |')
        str = f"{tracker},{rgb_prec_score1:.1f}, {rgb_norm_prec_score1:.1f},{rgb_succ_score1:.1f}," \
              f"{rgb_prec_score2:.1f}, {rgb_norm_prec_score1:.2f},{rgb_succ_score2:.1f}\n"
        wstr+=str

    with open("cal_results/TrackResult.csv","w") as f:
        f.write(wstr)