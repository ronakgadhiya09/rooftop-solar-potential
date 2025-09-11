from yacs.config import CfgNode as CN

config = CN()

config.MODEL = CN()
config.MODEL.NAME = 'seg_hrnet_ocr'
config.MODEL.PRETRAINED = ''
config.MODEL.ALIGN_CORNERS = True

config.DATASET = CN()
config.DATASET.DATASET = 'india_dataset'
config.DATASET.ROOT = 'inria_dataset/twoChannels_in'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.VAL_SET = 'val'
config.DATASET.IMAGE_SIZE = [512, 512]
config.DATASET.NUM_CLASSES = 1

config.SOLVER = CN()
config.SOLVER.BATCH_SIZE = 20
config.SOLVER.BASE_LR = 0.0001
config.SOLVER.LR_SCHEDULER = 'poly'
config.SOLVER.MAX_EPOCHES = 100
config.SOLVER.WEIGHT_DECAY = 0.0001

config.LOSS = CN()
config.LOSS.USE_OHEM = False
config.LOSS.OHEM_THRESH = 0.9
config.LOSS.CLASS_BALANCE = True

config.OUTPUT_DIR = 'i_outputs'
config.LOG_DIR = 'log'