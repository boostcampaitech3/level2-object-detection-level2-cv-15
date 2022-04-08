import pandas as pd
import json

test_json_path = "/opt/ml/detection/dataset/test.json"
# submission_file_path = "submission_latest.csv"
submission_file_path = "/opt/ml/detection/baseline/mmdetection/work_dirs/output (19).csv"
# pseudolabel_file_path = '/opt/ml/detection/dataset/pseudo_train.json'
pseudolabel_file_path = "/opt/ml/detection/dataset/train_new_1.json"

with open(test_json_path, "r") as test_json:
    test_dict = json.load(test_json)

pseudo_dict = {}

pseudo_dict["info"] = test_dict["info"]
pseudo_dict["licenses"] = test_dict["licenses"]
pseudo_dict["images"] = test_dict["images"]
pseudo_dict["categories"] = test_dict["categories"]
pseudo_dict["annotations"] = test_dict["annotations"]

submission_file = pd.read_csv(submission_file_path)

cnt = 0 
for idx, row in submission_file.iterrows():
    if ' ' not in row  : continue
    # print(row)
    row_unit = row["PredictionString"].split(" ")
    for j in range(0, len(row_unit)-6, 6):
        if float(row_unit[j+1]) > 0.9:
            category_id = row_unit[j]
            image_id = idx
            bbox = map(float,[row_unit[j+2], row_unit[j+3],row_unit[j+4],row_unit[j+5]])
            is_crowd = 0
            q = {'image_id':image_id, "category_id":category_id, "area" : float(row_unit[j+4]) * float(row_unit[j+5]), "iscrowd": 0, "id":cnt, "bbox":bbox}
            pseudo_dict["annotations"].append(q)
            cnt += 1

with open(pseudolabel_file_path, 'w', encoding='utf-8') as f:
    json.dump(pseudo_dict, f, ensure_ascii=False, indent=4)
    
submission_file.head()