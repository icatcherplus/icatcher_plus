import pytest
import sys
sys.path.append("./reproduce/")
import requests
import shutil
import os
from PIL import Image, ImageChops, ImageStat
from io import BytesIO
import face_recognition 
from reproduce.face_rec import FaceRec

face_image_link = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=687&q=80"
failure_mode_image_link= "https://www.cdc.gov/healthypets/images/pets/cute-dog-headshot.jpg"

def download_file(url):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        filename = os.path.basename(url)
        with open('/tmp/%s' % filename, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
            
        return '/tmp/%s' % filename

def convert_bbox(bbox):
    """
    Converts face recognition bounding box to the pipeline's
    param bbox: bounding box in [top, right, bottom, left] 
    :return: bbox in [left, top, width, height]
    """
    top = bbox[0]
    right = bbox[1]
    bottom = bbox[2]
    left = bbox[3]
                    
    w = right - left
    h = top - bottom

    return [left, top, w, h]


def test_image_generation():
    """
    Test that an image can be generated given a bounding box and a frame/image file
    If the function works, the number of known faces will be exactly 1
    """

    generated_image = download_file(face_image_link)
    
    test_img = face_recognition.load_image_file(generated_image)

    fr = FaceRec()
    fr.get_ref_image(generated_image)
    
    locs = fr.facerec_check(test_img)

    frame = face_recognition.load_image_file(generated_image)
    
    test_bbox = locs

    fr = FaceRec()
    fr.generate_ref_image(test_bbox, frame)
    assert len(fr.known_faces) == 1


def test_get_ref_image():
    """
    Test that the image can be captured, will capture and log a "known face"
    If the function works, the number of known faces will be exactly 1.
    """
    fr = FaceRec()

    generated_image = download_file(face_image_link)
    fr.get_ref_image(generated_image)
    assert len(fr.known_faces) == 1


def test_facerec_pass():
    """
    Test that the face recognition works with known faces, will register a match with a known face
    If the function works, it should return the location of a matching face
    """
    fr = FaceRec()

    generated_image = download_file(face_image_link)
    fr.get_ref_image(generated_image)
    
    test_img = face_recognition.load_image_file(generated_image)
    locs = fr.facerec_check(test_img)

    assert locs != None


def test_facerec_fail():
    """
    Test that the face recognition works with known faces, will not register a match with an unknown face
    If the function works, the checker will return a None value if it doesn't find anything.
    """
    fr = FaceRec()

    generated_image = download_file(face_image_link)
    failure_image = download_file(failure_mode_image_link)
    fr.get_ref_image(generated_image)

    test_img = face_recognition.load_image_file(failure_image)
    locs = fr.facerec_check(test_img)
    assert locs == None

def test_face_selection():
    """
    Test the face selection method that utilizes face recognition
    """
    fr = FaceRec()
    generated_image = download_file(face_image_link)

    fr.get_ref_image(generated_image)
    test_img = face_recognition.load_image_file(generated_image)

    bbox = fr.facerec_check(test_img)

    selected_bbox = fr.select_face([bbox], test_img, tolerance=0.9)

    assert bbox == selected_bbox
