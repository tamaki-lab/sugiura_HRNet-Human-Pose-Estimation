import os
import os.path

import numpy as np
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from io import BytesIO
import requests
from torchvision import transforms
from torchvision.utils import save_image
from pycocotools.coco import COCO
import pickle
from tqdm import tqdm
import json
import torch


def main():
    def crop_image(img, bbox):
        x_min, y_min, width, height = map(int, bbox)

        # 画像をクロップ
        cropped_img = transforms.functional.crop(
            img, y_min, x_min, height, width
        )
        # print(bbox)
        # print(cropped_img.size())

        return cropped_img

    def save_dict_to_txt(data_dict, filename):
        with open(filename, 'w') as file:
            json.dump(data_dict, file, indent=4)

    def adjust_keypoints(bbox, keypoints):
        # bboxの先頭2つの値を整数に変換
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])

        adjusted_keypoints = []

        # keypointsを3つずつ処理
        for i in range(0, len(keypoints), 3):
            x = keypoints[i] - bbox[0]
            y = keypoints[i + 1] - bbox[1]
            visibility = keypoints[i + 2]

            # 0以下の値は0にする
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            adjusted_keypoints.extend([x, y, visibility])

        return adjusted_keypoints

    path = '/mnt/NAS-TVS872XT/dataset/COCO/annotations/person_keypoints_val2017.json'

    coco = COCO(path)
    ids = list(coco.imgs.keys())

    num_keypoints = [0] * 18

    for index in tqdm(range(len(ids))):
        try:
            img_id = ids[index]
            # img_id = 280891
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            coco_url = coco.loadImgs(img_id)[0]['coco_url']

            try:
                response = requests.get(coco_url)
                img = Image.open(BytesIO(response.content)).convert('RGB')
            except requests.exceptions.RequestException as e:
                print(f"Failed to download image {img_id}: {e}")
                continue

            transform = transforms.ToTensor()
            img = transform(img)

            # if index == 1:
            #     save_image(img, f'image{index}_{i}.png')
            #     save_image(cropped_img, f'cropped_image{index}_{i}.png')

            for i in range(len(anns)):
                try:
                    image_id = anns[i]['image_id']
                    num_keypoint = anns[i]['num_keypoints']

                    # imageの保存
                    os.makedirs(f'normal/image/{image_id}/default', exist_ok=True)
                    os.makedirs(f'normal/image/{image_id}/cropped', exist_ok=True)
                    cropped_img = crop_image(img, anns[i]['bbox'])
                    if i == 0:
                        save_image(img, f'normal/image/{image_id}/default/image{i}.png')
                    save_image(cropped_img, f'normal/image/{image_id}/cropped/image{i}.png')

                    os.makedirs(f'num_keypoints/crop_image/{str(num_keypoint)}/{image_id}/', exist_ok=True)
                    save_image(cropped_img, f'num_keypoints/crop_image/{str(num_keypoint)}/{image_id}/image{i}.png')

                    # pickleの保存
                    bbox = anns[i]['bbox']
                    keypoints = anns[i]['keypoints']
                    anns[i]['keypoints'] = adjust_keypoints(bbox, keypoints)

                    os.makedirs(f'normal/pickle/{image_id}', exist_ok=True)
                    with open(f'normal/pickle/{image_id}/data{i}.pkl', 'wb') as f:
                        pickle.dump((np.array(cropped_img.permute(1, 2, 0) * 255), anns[i]), f)

                    os.makedirs(f'num_keypoints/pickle/{str(num_keypoint)}/{image_id}/', exist_ok=True)
                    with open(f'num_keypoints/pickle/{str(num_keypoint)}/{image_id}/data{i}.pkl', 'wb') as f:
                        pickle.dump((np.array(cropped_img.permute(1, 2, 0) * 255), anns[i]), f)

                    num_keypoints[num_keypoint] += 1

                except Exception as e:
                    print(f"Failed to process annotation {i} for image {img_id}: {e}")
                    continue
            # break
        except Exception as e:
            print(f"Failed to process image {img_id}: {e}")
            continue

    data_dict = {index: value for index, value in enumerate(num_keypoints)}
    save_dict_to_txt(data_dict, 'data_dict2.txt')


if __name__ == '__main__':
    main()

