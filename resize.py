import os
import cv2
import imghdr
def resize_dir (directory):
    files_to_process = []
    if not os.path.exists(directory):
        raise
    # Extract list of files to process
    for root, dirs, files in os.walk(directory):
        # keep only images
        files = [
            file for file in files if ((imghdr.what(os.path.join(root, file)) is not None) and ("_fixed" not in file) and ("_original" not in file))]
        if len(files) == 0:
            continue
        # keep only files from experiment 2 or set8
        if not ("EXPERIMENT 2" in root or "SET8" in root):
            continue
        print files
        files_to_process = files_to_process + [os.path.join(root,f) for f in files]
    print files_to_process
    for f in files_to_process:

        print "Now processing: {}".format(f)
        img = cv2.imread(f)
        resized_img = cv2.resize(img, (0,0), fx=1521.0/1352, fy=1003.0/885) 
        filename,extension = os.path.splitext(f)
        #print filename+"_fixed"+extension
        cv2.imwrite(filename+extension,resized_img)
        #cv2.imwrite(filename+"_fixed"+extension,resized_img)
        #cv2.imwrite(filename+"_original"+extension,img)

if __name__ == "__main__":
    resize_dir('./example_data/to_resize/')
