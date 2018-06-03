import os
from lxml import etree, objectify

# Classes
# Pedestrian              1
# Person on vehicle       2
# Car                     3
# Bicycle                 4
# Motorbike               5
# Non motorized vehicle   6
# Static person           7
# Distractor              8
# Occluder                9
# Occluder on the ground  10
# Occluder full           11
# Reflection              12


# gt.txt
# 1 Frame number Indicate at which frame the object is present
# 2 Identity number Each pedestrian trajectory is identified by a unique ID (1 for detections)
# 3 Bounding box left Coordinate of the top-left corner of the pedestrian bounding box
# 4 Bounding box top Coordinate of the top-left corner of the pedestrian bounding box
# 5 Bounding box width Width in pixels of the pedestrian bounding box
# 6 Bounding box height Height in pixels of the pedestrian bounding box
# 7 Confidence score DET: Indicates how confident the detector is that this instance is a pedestrian.
#   GT: It acts as a flag whether the entry is to be considered (1) or ignored (0).
# 8 Class GT: Indicates the type of object annotated
# 9 Visibility GT: Visibility ratio, a number between 0 and 1 that says how much of that object is visible. Can be due
#   to occlusion and due to image border cropping.

def root(folder, filename, width, height):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
            E.folder(folder),
            E.filename(filename),
            E.source(
                E.database('MOT17'),
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
    anns = []
    ann_dirs = ['data/MOT17/MOT17DetLabels/train/', 'data/MOT17/MOT17DetLabels/test/']
    for ann_dir in ann_dirs:
        for (dirpath, dirnames, filenames) in os.walk(ann_dir):
            if len(filenames)==0:
                continue
            for filename in sorted(filenames):
                if filename.endswith('.ini')==True:
                    ann      = {}
                    seq_info = dirpath + '/' + filename
                    with open(seq_info) as f:
                        lines         = f.readlines()
                        ann['folder'] = lines[1].rstrip('\n').split('=')[-1]
                        ann['imdir']  = lines[2].rstrip('\n').split('=')[-1]
                        ann['length'] = lines[4].rstrip('\n').split('=')[-1]
                        ann['width']  = lines[5].rstrip('\n').split('=')[-1]
                        ann['height'] = lines[6].rstrip('\n').split('=')[-1]
                        ann['imext']  = lines[7].rstrip('\n').split('=')[-1]
                    anns.append(ann)

        for ann in anns:
            xml_data = {}
            gt_path  = ann_dir + ann['folder'] + '/gt/gt.txt'
            with open(gt_path) as f:
                lines = f.readlines()
                for line in lines:
                    frame, tid, xmin, ymin, width, height, score, class_id, visibility = line.rstrip('\n').split(',')
                    if frame not in xml_data:
                        xml_data[frame] = []
                    obj = {}
                    obj['trackid'] = tid
                    obj['xmin']    = xmin
                    obj['ymin']    = ymin
                    obj['xmax']    = str(int(xmin) + int(width))
                    obj['ymax']    = str(int(ymin) + int(height))
                    obj['name']    = class_id
                    xml_data[frame].append(obj)

            count  = 1
            length = len(xml_data)
            for frame in sorted(xml_data.keys(), key = lambda x: int(x)):
                annotation = root(ann['folder'] + '/' + ann['imdir'], frame.zfill(6) + ann['imext'], ann['width'], ann['height'])
                for instance in xml_data[frame]:
                    annotation.append(instance_to_xml(instance))

                if ann_dir.split('/')[-2] == 'train':
                    if count<=((1-validation_split)*length):
                        if not os.path.isdir('data/MOT17Ann/train/' + ann['folder']):
                            os.makedirs('data/MOT17Ann/train/' + ann['folder'])
                        outfile = 'data/MOT17Ann/train/' + ann['folder'] + '/{}.xml'.format(frame.zfill(6))
                        etree.ElementTree(annotation).write(outfile, pretty_print=True)
                    else:
                        if not os.path.isdir('data/MOT17Ann/val/' + ann['folder']):
                            os.makedirs('data/MOT17Ann/val/' + ann['folder'])
                        outfile = 'data/MOT17Ann/val/' + ann['folder'] + '/{}.xml'.format(frame.zfill(6))
                        etree.ElementTree(annotation).write(outfile, pretty_print=True)
                    count += 1
                else:
                    if not os.path.isdir('data/MOT17Ann/test/' + ann['folder']):
                        os.makedirs('data/MOT17Ann/test/' + ann['folder'])
                    outfile = 'data/MOT17Ann/test/' + ann['folder'] + '/{}.xml'.format(frame.zfill(6))
                    etree.ElementTree(annotation).write(outfile, pretty_print=True)

if __name__=="__main__":

    validation_split = 0.25
    create_annotations(validation_split)
