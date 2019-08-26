import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def view_instance(img, gt_label, pr_label=None):
    img = img.cpu()
    # gt_label = gt_label.cpu()
    # pr_label = pr_label.cpu()
    # c, h, w = img.shape
    # with open('/home/spmatteo/MSDocuments/QuantLab/COCO/coco.names', 'r') as f:
    #     classes = [line.strip() for line in f.read().splitlines()]
    # cmap = plt.get_cmap('tab20b')
    # colors = [cmap(i) for i in np.linspace(0, 1, len(classes)-1)]
    # fig, ax = plt.subplots(1, figsize=(12, 9))
    # ax.imshow(img.permute(1, 2, 0))  # h, w, c
    # # browse annotations and draw bounding boxes
    # bboxes = []
    # if label is not None:
    #     for i, annotation in enumerate(label):
    #         cls = annotation[6]
    #         if i < 6:
    #             print(annotation, classes[int(cls)])
    #         color = colors[int(cls)]
    #         bbox = patches.Rectangle((annotation[0], annotation[1]), annotation[2]-annotation[0], annotation[3]-annotation[1],
    #                                  linewidth=2, edgecolor=color, facecolor='none', label=classes[int(cls)])
    #         ax.add_patch(bbox)
    #         bboxes.append((bbox, classes[int(cls)], color))
    # for bbox in bboxes:
    #     ax.annotate(bbox[1], bbox[0].get_xy(), weight='bold', fontsize=10, color=bbox[2])
    # plt.axis('off')
    # plt.show()
