import utils.det_utils as util
import scripts.train as trainer

data_dirs = ['Human2','Human3','Human4','Human5','Human6','Human7','Human8','Human9','Woman','Jogging-1','Jogging-2','Walking','Walking2']
val_data_dirs=['Human3','Human8','Jogging-2','Walking2']

frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = util.prepare_data()
# util.process_data(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs) #save features to file
x_train, y_train, x_val, y_val = util.get_trainval_data()
trainer.train(x_train, y_train, x_val, y_val)
