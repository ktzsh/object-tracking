import utils.det_utils as util
import scripts.train as trainer

data_dirs = ['Human2','Human3','Human4','Human5','Human6','Human7','Human8','Human9','Woman','Jogging-1','Jogging-2','Walking','Walking2', 'Biker','Subway', 'Skater', 'Singer1', 'Jumping', 'Jump']
val_data_dirs=['Human3','Human8','Jogging-2','Walking2', 'Skater', 'Jump']

# frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = util.prepare_data(data_dirs)
# # util.process_data(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs, data_dirs) #save features to file
# x_train_vis, x_train_heat, y_train, x_val_vis, x_val_heat, y_val = util.get_trainval_data(data_dirs, val_data_dirs)
# trainer.train(x_train_vis, x_train_heat, y_train, x_val_vis, x_val_heat, y_val)

# Simple Model
frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = util.prepare_data(data_dirs)
util.process_data_simple(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs, data_dirs) #save features to file
x_train_vis, x_train_heat, y_train, x_val_vis, x_val_heat, y_val = util.get_trainval_data_simple(data_dirs, val_data_dirs)
trainer.train_simple(x_train_vis, x_train_heat, y_train, x_val_vis, x_val_heat, y_val)
