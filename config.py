# @https://github.com/fire717/Fire

cfg = {
    ### Global Set
    "model_name": "dense",  
    #dense/swin
    'GPU_ID': '0',  #'' for cpu
    "class_number": 5990,

    "random_seed":42,
    "cfg_verbose":True,
    "num_workers":4,


    ### Train Setting
    'dict_path':"../data/char_std_5990.txt",

    'img_dir':"../data/images",
    'train_label_path': "../data/data_train_part.txt",# if 'DIR' quale  train_path
    'val_label_path':"../data/data_test_part.txt",
    'pretrained':'', #path or '' output/densenet_e2_0.85521.pth
    'log_interval':10,  
    'try_to_train_items': 0,   # 0 means all
    'save_best_only': True,  #only save model if better than before
    'save_one_only':True,    #only save one best model (will del model before)
    "save_dir": "output/",
    'pin_memory': True,
    'metrics': ['acc'], # default is acc,  can add F1  ...
    "loss": 'CTC', # default or '' means CE, can other be Focalloss-1, BCE...

    "load_in_mem": True, # load all img in mem

    ### Train Hyperparameters
    "img_size": [32, 280], # [h, w] 
    'learning_rate':0.0005,
    'batch_size':128,
    'epochs':10,
    'optimizer':'Adam',  #Adam  SGD 
    'scheduler':'step-1-0.4', #default  SGDR-5-2  CVPR   step-1-0.4

    'warmup_epoch':0, # 
    'weight_decay' : 0.000,
    # "k_flod":5,
    # 'start_fold':0,
    'early_stop_patient':16,

    # 'use_distill':0,
    # 'label_smooth':0,
    'checkpoint':None,
    # 'class_weight': None,#s[1.4, 0.78], # None [1, 1]
    'clip_gradient': 0,#1,       # 0
    # 'freeze_nonlinear_epoch':0,

    'dropout':0., #before last_linear

    # 'mixup':False,
    # 'sample_weights':None,


    ### Test
    'show_heatmap':False,
    
    'model_path':'output/dense_e69_0.83106.pth',#test model

    'test_img_dir':"../../data/challange/images",#test with label,get test acc
    'test_label_path':"../../data/challange/data_test.txt",

    'pre_img_dir':"../data/challange/test/",#test without label, just show img result
    'use_TTA':0,
    'test_batch_size': 1,
    

    ### log
    'comment':"",
    'log_item':["model_name",
                "img_size",
                "learning_rate",
                "batch_size",
                "epochs",
                "optimizer",
                "scheduler",
                "warmup_epoch",
                "weight_decay",
                # "k_flod",
                # "start_fold",
                # 'label_smooth',
                # 'class_weight',
                'clip_gradient',
                'dropout',
                'loss',
                'comment'],
    }
