# @https://github.com/fire717/Fire

cfg = {
    ### Global Set
    "model_name": "densenet",  
    #
    'GPU_ID': '1',
    "class_number": 5989,

    "random_seed":42,
    "cfg_verbose":True,
    "num_workers":4,


    ### Train Setting
    'dict_path':"../../data/char_std_5990.txt",

    'img_dir':"../../data/images",
    'train_label_path': "../../data/data_train.txt",# if 'DIR' quale  train_path
    'val_label_path':"../../data/data_test.txt",
    'pretrained':'output/densenet_e2_0.85521.pth', #path or ''
    'log_interval':10,  
    'try_to_train_items': 0,   # 0 means all
    'save_best_only': True,  #only save model if better than before
    'save_one_only':True,    #only save one best model (will del model before)
    "save_dir": "output/",
    'pin_memory': True,
    'metrics': ['acc'], # default is acc,  can add F1  ...
    "loss": 'CTC', # default or '' means CE, can other be Focalloss-1, BCE...


    ### Train Hyperparameters
    "img_size": [32, 280], # [h, w]
    'learning_rate':0.01,
    'batch_size':128,
    'epochs':20,
    'optimizer':'SGD',  #Adam  SGD 
    'scheduler':'default-0.1-3', #default  SGDR-5-2  CVPR   step-4-0.8

    'warmup_epoch':0, # 
    'weight_decay' : 0.0001,
    # "k_flod":5,
    # 'start_fold':0,
    'early_stop_patient':4,

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
    
    'model_path':'output/densenet_e2_0.85521.pth',#test model

    'eval_path':"./data/test",#test with label,get test acc
    'test_path':"../../data/test",#test without label, just show img result
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
