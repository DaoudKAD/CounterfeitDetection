import cv2
import os


def apply_mask(path_to_dataset, type):

    directory = os.fsencode(path_to_dataset+type+"/")
    for img_filename in os.listdir(directory):
        img_filename = img_filename.decode('utf-8')
        img1 = cv2.imread(path_to_dataset+type+"/"+img_filename)
        mask = cv2.imread(path_to_dataset+"mask_"+type+"/"+img_filename, 0)
        image_masked = cv2.bitwise_and(img1, img1, mask=mask)

        cv2.imwrite(path_to_dataset+type+"/"+img_filename, image_masked)
        print(img_filename+" : masked !")

if __name__ == '__main__':
    path = "/Users/daoud.kadoch/Documents/counterfeit-detection-with-cnn/test/"
    type = "scan"
    apply_mask(path, type)

