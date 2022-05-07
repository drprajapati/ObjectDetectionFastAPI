import cv2
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox

bounding = True


def detect_and_draw(filename, confidence=0.5):
    image_path = f"images/{filename}"
    image = cv2.imread(image_path)
    # bbox, conf = cv.detect_face(image, threshold=0.5, enable_gpu=False)
    # print(bbox, conf)
    # print(f"========================\nImage processed: {filename}\n")
    #
    # # output_image = draw_bbox(image, bbox=bbox, labels=[], confidence=conf)
    # if len(bbox) > 0:
    #     for idx, f in enumerate(bbox):
    #         rectified_f = [int(i * 100 / 20) for i in f]
    #
    #         (startX, startY) = rectified_f[0], rectified_f[1]
    #         (endX, endY) = rectified_f[2], rectified_f[3]
    #
    #         # draw rectangle over face
    #         if bounding == True:
    #             print(image,(startX, startY))
    #             bb_image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    #             cv2.imshow("image", bb_image)
    #             cv2.waitKey()
    # cv2.imshow("object_detection", output_image)
    # cv2.waitKey()
    # cv2.imwrite(f"results/bb_{filename}", output_image)
    bbox, label, conf = cv.detect_common_objects(image)

    output_image = draw_bbox(image, bbox, label, conf)
    cv2.imshow("object_detection", output_image)
    cv2.waitKey()


detect_and_draw('apple.jpeg')
