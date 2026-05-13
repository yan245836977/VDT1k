# Visual object tracking via integrating images of visible, thermal, and depth modalities

VDT1k includes 1,325 video sequences captured by a triple-modal camera mounted on a moving vehicle. This dataset is divided into 1,176 and 149 sequences for training and testing, respectively. These sequences consist of 240k aligned image pairs covering 16 types of objects photographed in motion from highways and downtown streets. Moreover, seven challenging attributes are annotated with binary labels, serving as crucial references for evaluating the performance of tracking methods under extreme scenarios. Welcome to use our dataset and cite our work!
## Instances
![You can play this video via clicking this image](https://i-blog.csdnimg.cn/direct/4edc9d8e1bc747a1b7cb2ad4a59e0244.png#pic_center)

You can play this video via clicking this [link](https://www.bilibili.com/video/BV1npmXBNEF3/?spm_id_from=333.1387.homepage.video_card.click&vd_source=c48289e4d0b5d554d90e058fce4c4388).

# How to use VDT1k
You can access related sources of VDT1k via BaiduNet link ().
Folder architecture of VDT1k:

```python
root:
	-annotations
	-sequences
	-test_list.txt
	-train_list.txt
```
We also provide evaluated toolkit in this repository. You should save your results in this format:

```python
root:
	-cal_results
	-cal_results_atts
	-VDT1k # Putting tracking results in here.
		- tracker # Your or compared method name.
			- rgbd # Tracking results based on RGBD modalities.
			- rgbt # Tracking results based on RGBT modalities.
	-att_rec.json
	-cal_all_seqs.py
	-cal_attributes.py
	-cal_iou_prec.py
```

You can obtain overview evaluation by running :
> cal_all_seqs.py

```python
if __name__ == '__main__':
    trakcers = os.listdir("./VDT1k")

    wstr = "Methods, PR(RGBT), NPR(RGBT), SR(RGBT), PR(RGBD), NPR(RGBD), SR(RGBD)\n"
    for tracker in trakcers:
        # rgbd
        pd_rt = f"./VDT1k/{tracker}/rgbd/"
        gt_rt = "F:/VDT_dataset/annotations/"

        seqs = [i.replace("\n","") for i in open(r"F:\VDT_dataset\test_list.txt").readlines()]
        rgb_succ_score_all1, rgb_prec_score_all1, rgb_norm_prec_score_all1 = [], [], []
```

You should put tracking results of your method and compared methods into "./VDT1k" before you run "cal_all_seqs.py"
Meanwhile, evaluated results will be recorded in .csv and saved in "./cal_results/TrackResult.csv"

Similarly, you can obtain attribute-based evaluation by running:
> cal_attributes.py

```python
if __name__ == '__main__':
    trakcers = os.listdir("./VDT1k")
    modality = "rgbt"
    with open("att_rec.json","r") as f:
        json_rec = json.load(f)
        att_names = ['OE', 'LI', 'LF', 'NI', 'SE', 'OL', 'OC'] #list(json_rec.keys())

    with open("att_rec.json","r") as f:
        attributes = json.load(f)
```
