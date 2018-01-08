# http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
# labels manually decided
LABELS =    [
                'Person', 'Face', 'Bird', 'Car', 'Deer', 'Dog', 'Bike', 'Panda'
            ]
#
# tb data dirs map
LABELS_DIR_MAP =    {
                        'Basketball'   : 'Person',
                        'Biker'        : 'Face',
                        'Bird1'        : 'Bird',
                        'Bird2'        : 'Bird',
                        'BlurBody'     : 'Person',
                        'BlurCar1'     : 'Car',
                        'BlurCar2'     : 'Car',
                        'BlurCar3'     : 'Car',
                        'BlurCar4'     : 'Car',
                        'BlurFace'     : 'Face',
                        'Bolt'         : 'Person',
                        'Bolt2'        : 'Person',
                        'Boy'          : 'Face',
                        'Car1'         : 'Car',
                        'Car2'         : 'Car',
                        'Car24'        : 'Car',
                        'Car4'         : 'Car',
                        'CarDark'      : 'Car',
                        'CarScale'     : 'Car',
                        'Couple'       : 'Person',
                        'Crossing'     : 'Person',
                        'Crowds'       : 'Person',
                        'Dancer2'      : 'Person',
                        'David'        : 'Face',
                        'David2'       : 'Face',
                        'David3'       : 'Person',
                        'Deer'         : 'Deer',
                        'Dog'          : 'Dog',
                        'Dudek'        : 'Face',
                        'FaceOcc1'     : 'Face',
                        'FaceOcc2'     : 'Face',
                        'FleetFace'    : 'Face',
                        'Football'     : 'Face',
                        'Freeman1'     : 'Face',
                        'Freeman4'     : 'Face',
                        'Girl'         : 'Face',
                        'Girl2'        : 'Person',
                        'Gym'          : 'Person',
                        'Human2'       : 'Person',
                        'Human3'       : 'Person',
                        'Human4'       : 'Person',
                        'Human5'       : 'Person',
                        'Human6'       : 'Person',
                        'Human7'       : 'Person',
                        'Human8'       : 'Person',
                        'Human9'       : 'Person',
                        'Jogging'      : 'Person',
                        'Jump'         : 'Person',
                        'Jumping'      : 'Face',
                        'KiteSurf'     : 'Face',
                        'Man'          : 'Face',
                        'Matrix'       : 'Face',
                        'Mhyang'       : 'Face',
                        'MotorRolling' : 'Bike',
                        'MountainBike' : 'Bike',
                        'Panda'        : 'Panda',
                        'Shaking'      : 'Face',
                        'Singer1'      : 'Person',
                        'Skater'       : 'Person',
                        'Skater2'      : 'Person',
                        'Skating1'     : 'Person',
                        'Skating2'     : 'Person',
                        'Soccer'       : 'Face',
                        'Subway'       : 'Person',
                        'Surfer'       : 'Face',
                        'Suv'          : 'Car',
                        'Trellis'      : 'Face',
                        'Walking'      : 'Person',
                        'Walking2'     : 'Person',
                        'Woman'        : 'Person'
                    }

# 640 * 360 px
import os
import cv2
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
    ann_dir    = 'data/VisualTrackingBenchmark/'

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
            gt_path  = 'data/VisualTrackingBenchmark/' + ann['folder'] + '/' + gt
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
                if not os.path.isdir('data/VisualTBAnn/train/' + ann['folder']):
                    os.makedirs('data/VisualTBAnn/train/' + ann['folder'])
                outfile = 'data/VisualTBAnn/train/' + ann['folder'] + '/{}.xml'.format(frame.zfill(4))
                etree.ElementTree(annotation).write(outfile, pretty_print=True)
            else:
                if not os.path.isdir('data/VisualTBAnn/val/' + ann['folder']):
                    os.makedirs('data/VisualTBAnn/val/' + ann['folder'])
                outfile = 'data/VisualTBAnn/val/' + ann['folder'] + '/{}.xml'.format(frame.zfill(4))
                etree.ElementTree(annotation).write(outfile, pretty_print=True)
            count += 1

if __name__=="__main__":

    validation_split = 0.25
    create_annotations(validation_split)
