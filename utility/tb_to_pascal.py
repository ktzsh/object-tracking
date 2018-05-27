# http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

# 640 * 360 px
import os
import cv2
import json
from lxml import etree, objectify

def root(folder, filename, width, height):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
            E.folder(folder),
            E.filename(filename),
            E.source(
                E.database('VisualTB'),
                ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(3),
                )
    )

def instance_to_xml(obj):
    E = objectify.ElementMaker(annotate=False)
    return E.object(
            E.name(obj['name']),
            E.trackid(obj['trackid']),
            E.bndbox(
                E.xmin(obj['xmin']),
                E.ymin(obj['ymin']),
                E.xmax(obj['xmax']),
                E.ymax(obj['ymax']),
                )
    )

def create_annotations(validation_split):
    anns       = []
    anns_dict  = {}
    exclusions = ['panda-all.txt']

    with open("config.json") as config_buffer:
        config = json.loads(config_buffer.read())

    ann_dir        = config['train']['train_image_folder']
    LABELS_DIR_MAP = config['classes']['classes_map']

    # for dirs where numbering for frames does not start from 0001.jpg
    start_frame =   {
                        'BlurCar1' : 247,
                        'BlurCar3' : 3,
                        'BlurCar4' : 18,
                    }

    # http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
    # gt starts and end from these frames even though img exists
    skip_map =  {
                    'David'    : [300, 770],
                    'Freeman4' : [1, 283]
                }

    for (dirpath, dirnames, filenames) in os.walk(ann_dir):
        if len(filenames)==0:
            continue
        for filename in sorted(filenames):
            if filename in exclusions:
                continue
            if filename.startswith('._')==True:
                continue
            if filename.endswith('.txt')==True:
                folder = dirpath.split('/')[-1]

                try:
                    frame = dirpath + '/img/0001.jpg'
                    if folder in start_frame:
                        frame = dirpath + '/img/' + str(start_frame[folder]).zfill(4) + '.jpg'
                    image = cv2.imread(frame)
                    h, w, c = image.shape
                except:
                    print 'ERROR:', filename, frame
                    continue

                if folder in anns_dict:
                    anns[ anns_dict[folder] ]['gt'] += [filename]
                    continue
                ann      = {}
                ann['folder'] = folder
                ann['gt']     = [filename]
                ann['imdir']  = 'img'
                ann['width']  = w
                ann['height'] = h
                ann['imext']  = '.jpg'
                anns.append(ann)
                anns_dict[folder] = len(anns) - 1

    for ann in anns:
        xml_data = {}
        for idx, gt in enumerate(ann['gt']):
            gt_path  = ann_dir + ann['folder'] + '/' + gt
            with open(gt_path) as f:
                lines = f.readlines()
                frame = 1
                if ann['folder'] in start_frame:
                    frame = start_frame[ann['folder']]
                for line in lines:
                    if ann['folder'] in skip_map:
                        if (frame<skip_map[ ann['folder'] ][0]) or (frame>skip_map[ ann['folder'] ][1]):
                            frame += 1
                            continue
                    try:
                        xmin, ymin, width, height = line.rstrip('\n').split(',')
                    except:
                        xmin, ymin, width, height = line.rstrip('\n').split()

                    if str(frame) not in xml_data:
                        xml_data[str(frame)] = []

                    obj            = {}
                    obj['trackid'] = idx
                    obj['xmin']    = xmin
                    obj['ymin']    = ymin
                    obj['xmax']    = str(int(xmin) + int(width))
                    obj['ymax']    = str(int(ymin) + int(height))
                    obj['name']    = LABELS_DIR_MAP[ann['folder']]
                    xml_data[str(frame)].append(obj)
                    frame += 1

        count  = 1
        length = len(xml_data)
        for frame in sorted(xml_data.keys(), key = lambda x: int(x)):
            annotation = root(ann['folder'] + '/' + ann['imdir'], frame.zfill(4) + ann['imext'], ann['width'], ann['height'])
            for instance in xml_data[frame]:
                annotation.append(instance_to_xml(instance))

            if count<=((1-validation_split)*length):
                path = config['train']['train_annot_folder'] + ann['folder']
                if not os.path.isdir(path):
                    os.makedirs(path)
                outfile = path + '/{}.xml'.format(frame.zfill(4))
                etree.ElementTree(annotation).write(outfile, pretty_print=True)
            else:
                path = config['val']['val_annot_folder'] + ann['folder']
                if not os.path.isdir(path):
                    os.makedirs(path)
                outfile = path + '/{}.xml'.format(frame.zfill(4))
                etree.ElementTree(annotation).write(outfile, pretty_print=True)
            count += 1

if __name__=="__main__":

    validation_split = 0.25
    create_annotations(validation_split)
