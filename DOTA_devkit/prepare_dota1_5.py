import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO import DOTA2COCOTest, DOTA2COCOTrain
import argparse

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
               'helicopter', 'container-crane']

class_names = ["Boeing737", "Boeing777", "Boeing747", "Boeing787", "A320", "A220", "A330",
               "A350", "A321", "C919", "ARJ21", "other-airplane", "Passenger Ship", "motorboat", "fishing boat",
               "tugboat",
               "engineering ship", "liquid cargo ship", "Dry Cargo Ship", "warship", "other-ship", "small car", "bus",
               "cargo truck",
               "dump truck", "van", "trailer", "tractor", "truck tractor", "excavator", "other-vehicle",
               "baseball field",
               "basketball court", "football field", "tennis court", "roundabout", "intersection", "bridge"
               ]

airplanes = [
    "Boeing737", "Boeing777", "Boeing747", "Boeing787", "A320", "A220",
    "A330", "A350", "A321", "C919", "ARJ21", "other-airplane"
]

ships = [
    "Passenger Ship", "motorboat", "fishing boat", "tugboat",
    "engineering ship", "liquid cargo ship", "Dry Cargo Ship", "warship", "other-ship"
]

vehicles = [
    "small car", "bus", "cargo truck", "dump truck", "van", "trailer",
    "tractor", "truck tractor", "excavator", "other-vehicle"
]

courts = [
    "baseball field", "basketball court", "football field", "tennis court"
]

road = [
    "roundabout", "intersection", "bridge"
]

class_map = {}


def expand_class_map(subclass_names, class_name):
    for subclass_name in subclass_names:
        class_map[subclass_name.lower()] = class_name


expand_class_map(airplanes, "airplane")
expand_class_map(ships, "ship")
expand_class_map(vehicles, "vehicle")
expand_class_map(courts, "court")
expand_class_map(road, "road")

class_names = ["airplane", "ship", "vehicle", "court", "road"]


def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota1')
    parser.add_argument('--srcpath', default='gaofen_data')
    parser.add_argument('--dstpath', default=r'gaofen_data/fair1_1000',
                        help='prepare data')
    args = parser.parse_args()

    return args


def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)


def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)


def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)


def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)


def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')


def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """
    sub_size = 1000
    testPath = os.path.join(dstpath, f'test{sub_size}')
    trainPath = os.path.join(dstpath, f'train{sub_size}')
    valPath = os.path.join(dstpath, f'val{sub_size}')
    if not os.path.exists(testPath):
        os.makedirs(testPath)
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    if not os.path.exists(valPath):
        os.makedirs(valPath)

    # split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
    #                                                trainPath,
    #                                                gap=200,
    #                                                subsize=sub_size,
    #                                                num_process=32
    #                                                )
    # split_train.splitdata(1)
    #
    # split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
    #                                              valPath,
    #                                              gap=200,
    #                                              subsize=sub_size,
    #                                              num_process=32
    #                                              )
    # split_val.splitdata(1)
    #
    # split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test', 'images'),
    #                                                     os.path.join(testPath, 'images'),
    #                                                     gap=200,
    #                                                     subsize=sub_size,
    #                                                     num_process=32
    #                                                     )
    # split_test.splitdata(1)

    DOTA2COCOTrain(valPath, os.path.join(valPath, f'val{sub_size}_5classes.json'), class_map, class_names,
                   difficult='-1')
    DOTA2COCOTrain(trainPath, os.path.join(trainPath, f'train{sub_size}_5classes.json'), class_map, class_names,
                   difficult='-1')
    # DOTA2COCOTest(testPath, os.path.join(testPath, f'test{sub_size}_5classes.json'), class_map, class_names)


if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)
