import os, sys

if __name__ == "__main__":
    # both folders must have same number of files, with THE SAME NAMES (.seg)
    path_seg_1 = sys.argv[1]
    path_seg_2 = sys.argv[2]

    for (_, _, filenames) in walk(path_seg_1):
        for filename in filenames:
            os.system("./utils/ISM/Segmentations_metrics "+path_seg_1+"/"+filename+ " "+path_seg_2+"/"+filename)
