import os
import cv2
import copy
import json
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, normalize, bbox_iou, generate_heatmap_feat

def parse_annotation(ann_dir, img_dir, labels=[]):
    anns        = []
    all_imgs    = []
    seen_labels = {}

    # anns = sorted(os.listdir(ann_dir))
    for (dirpath, dirnames, filenames) in os.walk(ann_dir):
        # if len(anns)>=256:
        #     break
        if len(filenames)==0:
            continue
        for filename in sorted(filenames):
            if filename.endswith('.xml')==True:
                anns.append(dirpath+'/'+filename)

    for ann in anns:
        img = {'object':[]}

        # tree = ET.parse(ann_dir + ann)
        tree = ET.parse(ann)

        folder = ''
        for elem in tree.iter():
            if 'folder' in elem.tag:
                folder = elem.text + '/'
                img['folder'] = folder
            if 'filename' in elem.tag:
                img['filename'] = img_dir + folder + elem.text
                if '.' not in img['filename']:
                    img['filename'] += '.JPEG'
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels

def create_sequences_from_parsed_annotations(parsed_data, SEQUENCE_LENGTH):
    all_sequences = []
    n = len(parsed_data) - SEQUENCE_LENGTH + 1
    i = 0
    for i in range(n):
        j = i + SEQUENCE_LENGTH
        while parsed_data[i]['folder'] != parsed_data[i+SEQUENCE_LENGTH-1]['folder']:
            i = i + 1
            j = j + 1
        all_sequences.append(parsed_data[i:j])
    return all_sequences

class BaseBatchGenerator(Sequence):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       augment=True,
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.augment = augment
        self.norm    = norm

        self.counter = 0

        with open("config.json") as config_buffer:
            self.debug = json.loads(config_buffer.read())["train"]["debug"]

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                sometimes(iaa.Affine(
                )),
                iaa.SomeOf((0, 4),
                    [
                        iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 2.0
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.8, 1.2), per_channel=0.5),
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def aug_image(self, train_instance, augment, resize=True):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        h, w, c = image.shape

        all_objs = copy.deepcopy(train_instance['object'])

        if augment:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        if resize: image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if augment: obj[attr] = int(obj[attr] * scale - offx)
                if resize:
                    obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                    obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                if augment: obj[attr] = int(obj[attr] * scale - offy)
                if resize:
                    obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                    obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if augment and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin
        return image, all_objs

    def __getitem__(self, idx):
        raise NotImplementedError

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)
        self.counter = 0


class BatchGenerator(BaseBatchGenerator):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       augment=True,
                       norm=None):

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

        super(BatchGenerator, self).__init__(  images,
                                               config,
                                               shuffle=shuffle,
                                               augment=augment,
                                               norm=norm)

    def output_from_instance(self, train_instance, idx):
        x_instance = np.zeros((self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        b_instance = np.zeros((1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))
        y_instance = np.zeros((self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))

        # augment input image and fix object's position and size
        img, all_objs = self.aug_image(train_instance, augment=self.augment)

        # construct output from object's x, y, w, h
        true_box_index = 0

        for obj in all_objs:

            if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                center_x = .5*(obj['xmin'] + obj['xmax'])
                center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                center_y = .5*(obj['ymin'] + obj['ymax'])
                center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                    obj_indx  = self.config['LABELS'].index(obj['name'])

                    center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                    center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell

                    box = [center_x, center_y, center_w, center_h]

                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou     = -1

                    shifted_box = BoundBox(0,
                                           0,
                                           center_w,
                                           center_h)

                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou    = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor = i
                            max_iou     = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y
                    y_instance[grid_y, grid_x, best_anchor, 0:4] = box
                    y_instance[grid_y, grid_x, best_anchor, 4  ] = 1.
                    y_instance[grid_y, grid_x, best_anchor, 5+obj_indx] = 1

                    # assign the true box to b
                    b_instance[0, 0, 0, true_box_index] = box

                    true_box_index += 1
                    true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

        if self.debug:
            # plot image and bounding boxes for sanity check
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    cv2.putText(img[:,:,::-1], obj['name'],
                                (obj['xmin']+2, obj['ymin']+12),
                                0, 1.2e-3 * img.shape[0],
                                (0,255,0), 2)

            if not os.path.isdir('data/debug/' + str(idx)):
                os.makedirs('data/debug/' + str(idx))
            file_path = 'data/debug/' + str(idx) + '/' + train_instance['filename'].split('/')[-1]
            cv2.imwrite(file_path, img)

        # assign input image to x
        if self.norm != None:
            x_instance = self.norm(img)
        else:
            x_instance = img

        return [x_instance, b_instance], y_instance

    def __getitem__(self, idx):

        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        # input images
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))
        # desired network output
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))

        for train_instance in self.images[l_bound:r_bound]:
            [x_instance, b_instance], y_batch = self.output_from_instance(train_instance, idx)

            x_batch[instance_count] = x_instance
            b_batch[instance_count] = b_instance
            y_batch[instance_count] = y_instance
            instance_count          = instance_count + 1

        self.counter += 1
        return [x_batch, b_batch], y_batch


class BatchSequenceGenerator1(BatchGenerator):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       augment=True,
                       norm=None):

        self.seed  = 1
        images_seq = create_sequences_from_parsed_annotations(images, config['SEQUENCE_LENGTH'])
        print "Samples/Sequences", len(images), len(images_seq)
        super(BatchSequenceGenerator1, self).__init__(  images_seq,
                                                       config,
                                                       shuffle=shuffle,
                                                       augment=augment,
                                                       norm=norm)

    def __len__(self):
        return super(BatchSequenceGenerator1, self).__len__()

    def aug_image(self, train_instance, augment):
        # apply same augmentation to all images in batch
        np.random.seed(self.seed)
        return super(BatchSequenceGenerator1, self).aug_image(train_instance, augment)

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        batch_size = self.config['BATCH_SIZE']

        x_batch = np.zeros((batch_size, self.config['SEQUENCE_LENGTH'], self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        b_batch = np.zeros((batch_size, self.config['SEQUENCE_LENGTH'], 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))
        y_batch = np.zeros((batch_size, self.config['SEQUENCE_LENGTH'], self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))

        for i, train_seq in enumerate(self.images[l_bound:r_bound]):
            self.seed = np.random.randint(low=0, high=10000) + idx

            for j, train_instance in enumerate(train_seq):
                [x_instance, b_instance], y_instance = super(BatchSequenceGenerator1, self).output_from_instance(train_instance, idx)

                x_batch[i,j,:,:,:]     = x_instance
                b_batch[i,j,:,:,:,:,:] = b_instance
                y_batch[i,j,:,:,:,:]   = y_instance

        return [x_batch, b_batch], [y_batch, y_batch]

    def on_epoch_end(self):
        super(BatchSequenceGenerator1, self).on_epoch_end()


class BatchSequenceGenerator2(BaseBatchGenerator):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       augment=True,
                       norm=None):

        self.seed  = 1
        images_seq = create_sequences_from_parsed_annotations(images, config['SEQUENCE_LENGTH'])
        print "Sample Sequences", len(images_seq)
        super(BatchSequenceGenerator2, self).__init__( images_seq,
                                                       config,
                                                       shuffle=shuffle,
                                                       augment=augment,
                                                       norm=norm)

    def __len__(self):
        return super(BatchSequenceGenerator2, self).__len__()

    def aug_image(self, train_instance, augment):
        # apply same augmentation to all images in batch
        np.random.seed(self.seed)
        return super(BatchSequenceGenerator2, self).aug_image(train_instance, augment, resize=False)

    def output_from_instance(self, train_instance, idx):
        x_instance     = np.zeros((self.config['IMAGE_FV_H'], self.config['IMAGE_FV_W'], self.config['IMAGE_FV_C']))
        xbbox_instance = np.zeros(self.config['OUTPUT_SHAPE'])
        ybbox_instance = np.zeros(self.config['OUTPUT_SHAPE'])

        # augment input image and fix object's position and size
        img, objs = self.aug_image(train_instance, augment=self.augment)
        h, w, c   = img.shape

        if not os.path.isdir('data/tmp/' + str(idx)):
            os.makedirs('data/tmp/' + str(idx))
        file_path = 'data/tmp/' + str(idx) + '/' + train_instance['filename'].split('/')[-1]
        cv2.imwrite(file_path, img)
        # if self.debug: print "Aug image saved at..", file_path

        obj_det, vis_feat = self.config['DETECTOR'].extract_spatio_info(file_path, self.config['DETECTION_FV_LAYER'])
        vis_feat = vis_feat.reshape((self.config['IMAGE_FV_H'], self.config['IMAGE_FV_W'], self.config['IMAGE_FV_C']))

        # single object in groundtruth and from detection net
        if len(objs) != 1:
            if self.debug: print "[WARN] Multiple Objects in GT.........Choosing First One Only", idx
        if len(obj_det) != 1:
            if self.debug: print "[WARN] Multiple Objects in Detector...Choosing The One with highest prob and label match with GT", idx
            obj_det = [o for o in obj_det if o[0] == objs[0]['name'].lower()]

        # Center corodinates and width height
        det_x = ((objs[0]['xmin'] + objs[0]['xmax'])/2.0) / w
        det_y = ((objs[0]['ymin'] + objs[0]['ymax'])/2.0) / h
        det_w = ((objs[0]['xmax'] - objs[0]['xmin'])/2.0) / w
        det_h = ((objs[0]['ymax'] - objs[0]['ymin'])/2.0) / h

        if len(obj_det) != 0:
            # print "Detection succeded..", idx
            if self.debug: cv2.rectangle(img[:,:,::-1],
                          (int(obj_det[0][2][0] - obj_det[0][2][2]/2.0), int(obj_det[0][2][1] - obj_det[0][2][3]/2.0)),
                          (int(obj_det[0][2][0] + obj_det[0][2][2]/2.0), int(obj_det[0][2][1] + obj_det[0][2][3]/2.0)),
                          (0, 255, 0), 3)

            det_x_in = (obj_det[0][2][0]) / w
            det_y_in = (obj_det[0][2][1]) / h
            det_w_in = (obj_det[0][2][2]) / w
            det_h_in = (obj_det[0][2][3]) / h
        else:
            det_x_in = 0
            det_y_in = 0
            det_w_in = 0
            det_h_in = 0

        if self.config['OUTPUT_SHAPE'] == (4,):
            xbbox_instance = np.array([det_x_in, det_y_in, det_w_in, det_h_in], dtype='float32')
            ybbox_instance = np.array([det_x, det_y, det_w, det_h], dtype='float32')
        else:
            xbbox_instance = generate_heatmap_feat(det_x_in - det_w_in/2.0, det_y_in - det_h_in/2.0, det_w_in, det_h_in, hmap_size=self.config['HEATMAP_SIZE'])
            ybbox_instance = generate_heatmap_feat(det_x - det_w/2.0, det_y - det_h/2.0, det_w, det_h, hmap_size=self.config['HEATMAP_SIZE'])

        os.remove(file_path)
        if self.debug:
            # plot image and bounding boxes for sanity check
            for obj in objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    cv2.putText(img[:,:,::-1], obj['name'],
                                (obj['xmin']+2, obj['ymin']+12),
                                0, 1.2e-3 * img.shape[0],
                                (0,255,0), 2)

            if not os.path.isdir('data/debug/' + str(idx)):
                os.makedirs('data/debug/' + str(idx))
            file_path = 'data/debug/' + str(idx) + '/' + train_instance['filename'].split('/')[-1]
            cv2.imwrite(file_path, img)

        # assign input to x
        x_instance = vis_feat

        return [x_instance, xbbox_instance], ybbox_instance

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        batch_size = self.config['BATCH_SIZE']

        x_batch     = np.zeros((batch_size, self.config['SEQUENCE_LENGTH'], self.config['IMAGE_FV_H'], self.config['IMAGE_FV_W'], self.config['IMAGE_FV_C']))
        xbbox_batch = np.zeros((batch_size, self.config['SEQUENCE_LENGTH'],) + self.config['OUTPUT_SHAPE'])
        ybbox_batch = np.zeros((batch_size, self.config['SEQUENCE_LENGTH'],) + self.config['OUTPUT_SHAPE'])

        for i, train_seq in enumerate(self.images[l_bound:r_bound]):
            self.seed = np.random.randint(low=0, high=10000) + idx

            for j, train_instance in enumerate(train_seq):
                [x_instance, xbbox_instance], ybbox_instance = self.output_from_instance(train_instance, idx)

                x_batch[i,j]     = x_instance
                xbbox_batch[i,j] = xbbox_instance
                ybbox_batch[i,j] = ybbox_instance

        return [x_batch, xbbox_batch], ybbox_batch

    def on_epoch_end(self):
        os.rmdir('data/tmp/')
        os.rmdir('data/debug/')
        super(BatchSequenceGenerator2, self).on_epoch_end()
