import csv
import os.path

name = 'train'

annotation_file = 'oiddata/boxes/' + name + '-annotations-bbox.csv'
base_path = 'oiddata/' + name + '/'
path_to_put_locations = 'data/' + name + '_image_locations.txt'
path_to_put_annotations = 'data/' + name + '_annotations.csv'

class_labels = {'/m/03bj1': 'Panda', '/m/01lsmm': 'Scissors', '/m/078jl': 'Snake'}
class_ids = {'Panda': 1, 'Scissors': 2, 'Snake': 3}

annotations = []
locations = []


def in_annotations(imgid):
    for ann in annotations:
        if ann['imgid'] == imgid:
            return True
    return False


with open(annotation_file) as f:
    next(f)
    print('Creating annotations...', end=' ')
    lines = csv.reader(f, delimiter=',')
    for line in lines:
        img_id = line[0]
        class_label = line[2]
        # one bbox per image (this will make the model worse but less complicated)
        if class_label in class_labels:  # and not in_annotations(img_id):
            class_label = class_labels[class_label]
            path_to_file = base_path + class_label.lower() + '/' + img_id + '.jpg'
            if os.path.isfile(path_to_file):
                annotations.append({
                    'imgid': img_id,
                    'xmin': float(line[4]),
                    'ymin': float(line[6]),
                    'xmax': float(line[5]),
                    'ymax': float(line[7]),
                    'class': class_ids[class_label]
                })
                locations.append('src/' + path_to_file)

with open(path_to_put_locations, 'w') as loc_file:
    for line in locations:
        loc_file.write(line + '\n')

with open(path_to_put_annotations, 'w', newline='') as ann_file:
    writer = csv.DictWriter(ann_file, fieldnames=['imgid', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    writer.writeheader()
    for line in annotations:
        writer.writerow(line)

print('Files generated')
