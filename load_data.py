import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# JSONファイルのパス
json_file_path = 'data/coco/annotations/person_keypoints_train2017.json'

# 保存先ディレクトリ
save_dir = 'data/coco/images/train2017'

os.makedirs(save_dir, exist_ok=True)

# JSONファイルを読み込む
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 画像情報を取得
images = data['images']

def download_and_save_image(img_info):
    img_url = img_info['coco_url']
    file_name = img_info['file_name']

    try:
        # 画像をダウンロード
        response = requests.get(img_url)
        response.raise_for_status()

        # 画像をPILで開く
        img = Image.open(BytesIO(response.content))

        # 画像を保存
        img.save(f'{save_dir}/{file_name}')
        return f'Successfully saved {file_name}'
    except Exception as e:
        return f'Failed to download {file_name}: {e}'

# スレッドプールを使って並列に画像をダウンロード
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(download_and_save_image, img_info) for img_info in images]
    for future in tqdm(as_completed(futures), total=len(images)):
        print(future.result())
