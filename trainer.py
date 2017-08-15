import sys
import utils.det_utils as util
import scripts.model as model_h

data_dirs = ['Human2','Human3','Human4','Human5','Human6','Human7','Human8','Human9','Woman','Jogging-1','Jogging-2','Walking','Walking2', 'Biker','Subway', 'Skater', 'Singer1', 'Walking2', 'Jumping', 'Jump', 'Skating1', 'Girl', 'Girl2', 'Dancer']
val_data_dirs=['Human3','Human8','Jogging-2', 'Skater', 'Jump', 'Girl2', 'Dancer']


frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = util.prepare_data(data_dirs)
if sys.argv[1]=='extract':
    print "INFO: Feature Extraction Initiated.."
    if sys.argv[2]=='simple':
        print "INFO: Simple Model Chosen.."
        util.process_data_simple(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs, data_dirs) #save features to file
    elif sys.argv[2]=='normal':
        print "INFO: Deeper Model Chosen.."
        util.process_data(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs, data_dirs) #save features to file
elif sys.argv[1]=='train':
    print "INFO: Training on Subset Data TB-50 Initiated.."
    if sys.argv[2]=='simple':
        print "INFO: Simple Model Chosen.."
        x_train, y_train, x_val, y_val = util.get_trainval_data_simple(data_dirs, val_data_dirs)
        model_h.train_simple(x_train, y_train, x_val, y_val)
    elif sys.argv[2]=='normal':
        print "INFO: Deeper Model Chosen.."
        x_train_vis, x_train_heat, y_train, x_val_vis, x_val_heat, y_val = util.get_trainval_data(data_dirs, val_data_dirs)
        model_h.train(x_train_vis, x_train_heat, y_train, x_val_vis, x_val_heat, y_val)
