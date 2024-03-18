DATA = {
    'data_root': "/SSD2/dataset/Unsu-VOS",
    'pretrain': "DUTS_train",
    'best_pretrained_model': "./log/2024-03-17 14:02:19/model/best_model.pth",
    'DAVIS_train_main': "DAVIS_train",
    'DAVIS_train_sub': "YTVOS_train", # or None
    'DAVIS_val': "DAVIS_test",
}

TRAIN = {
    'GPU': "0, 1",
    'epoch': 200,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'img_size': 512,
    'slot_num': 2
}
