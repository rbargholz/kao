'''This program takes an image, finds the face and crops it to a set size'''

import glob
import os
import cv2
import PIL.Image as Image

#Original code offered by the GitHub user Onlyjus @ https://stackoverflow.com/users/979203/onlyjus
#Any modifications are done by me

def detect_face(image, face_cascade, return_img=False):
    min_size = (64, 64)
    min_neighbours = 3
    haar_flags = 0

    #This spreads pixels across the range, improving contrast
    #Betters final data result
    cv.equalizeHist(image, image)

    #Load faces
    faces = cv.HaarDetectObjects(
        image, face_cascade, cv.CreateMemStorage(0),
        haar_scale, min_neighbours, haar_flags, min_size)

    if(faces & return_img):
        for((x, y, width, height), n) in faces:
            #Construct a rectangle
            cv_point_1, cv_point_2 = (int(x) + int(y)), (int(x + w), int(y + h))
            cv.Rectangle(cv_point_1, cv_point_2, image, cv.RGB(255, 0, 0), 5, 8, 0)
    
    if return_img:
        return image
    else:
        return faces

def convert_to_greyscale(image_to_convert):
    image_to_convert = image_to_convert.convert('L')
    cv_image = cv2.cv.CreateImageHeader(image.size, cv2.IPL_DEPTH_8U, 1)
    cv.SetData(cv_image, image.tostring(), image.size[0])

    return cv_image

def cvimage_to_pilimage(cv_image):
    return Image.fromstring('L', cv.GetSize(cv_image), cv_image.tostring())

def crop_image(image, crop_box, box_scale=1):
    x_delta = max(crop_box[2] * (box_scale - 1), 0)
    y_delta = max(crop_box[3] * (box_scale - 1), 0)

    pil_box = [crop_box[0] - x_delta, crop_box[1] - y_delta, crop_box[0] + crop_box[2] + x_delta, crop_box[1] + crop_box[3] + y_delta]

    return image.crop(pil_box)

def face_crop(image_pattern, box_scale=1):
    face_cascade = cv.Load('haarcascade_frontalface_alt.xml')

    img_list = glob.glob(image_pattern)
    num_images = len(img_list)

    if num_images <= 0:
        print('No images in the list to process')
        return

    for img in img_list:
        pil_image = Image.open(img)
        cv_image = convert_to_greyscale(pil_image)

        faces = detect_face(cv_image, face_cascade)

        if faces:
            face_count = 1
            for face in faces:
                cropped_image = crop_image(pil_image, face[0], box_scale = box_scale)
                file_name, ext = os.path.splitext(img)

                cropped_image.save(file_name+'_facecrop_'+str(face_count)+ext)
                face_count += 1
        else:
            print('No faces found in', img)

def test(image_path):
    print("testing")
    print(cv2.__version__)
    pil_image = Image.open(image_path)
    cv_image = convert_to_greyscale(pil_image)

    face_cascade = cv2.Load('haarcascade_frontalface_alt.xml')
    face_image = detect_face(cv_image, face_cascade, return_image=True)

    img = cv_image_to_pilimage(face_image)

    img.show()
    img.save('test.jpg')

test('face_crop_test.jpg')
