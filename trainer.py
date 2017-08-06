import utils.det_utils as util
from scripts.train import *

frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = util.prepare_data()
util.process_data(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs)
