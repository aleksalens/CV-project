import cv2 as cv
import numpy as np
import cvzone

# Iniciranje video zapisivanja
video = cv.VideoCapture(0, cv.CAP_DSHOW)

video.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
video.set(cv.CAP_PROP_EXPOSURE, -4)


# Ucitavanje nove pozadinske slike
newBgImage = cv.imread("summer-beach-background.jpg")
if newBgImage is None:
    print("Error: Could not load background image.")
    exit()

# Zabeležavanje početnog frame-a kao referente pozadine
ret, bgReference = video.read()

def resize(dst, img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (width, height)
    resized = cv.resize(dst, dim, interpolation=cv.INTER_AREA)
    return resized

def adjust_brightness(image, brightness_increment=50, threshold=110):
    # Kreiramo masku koja pamti vrednosti piksela koji su veci od praga
    mask = image > threshold

    brightness_matrix = np.full(image.shape, brightness_increment, dtype=np.uint8)

    # Povecavamo vrednosti piksela za one piksele za koje je maska jednaka True
    brightened_image = np.where(mask, cv.add(image, brightness_matrix), image)

    return brightened_image


takeBgImage = 0

while True:
    ret, img = video.read()
    if not ret:
        print("Error: Could not read frame from video source.")
        break

    # Resize nove pozadine koja odgovara veličini video frejma
    if newBgImage.shape[:2] != bgReference.shape[:2]:
        bg = resize(newBgImage, bgReference)
    else:
        bg = newBgImage

    if takeBgImage == 0:
        bgReference = img
        cv.imshow("bgReference", bgReference)
        bgReference = adjust_brightness(bgReference)
        cv.imshow("adjusted brightness bgReference", bgReference)


    # Kreiranje maske
    diff1 = cv.subtract(img, bgReference)
    diff2 = cv.subtract(bgReference, img)

    cv.imshow("diff1", diff1)
    cv.imshow("diff2", diff2)

    diff = cv.add(diff1, diff2)
    diff[abs(diff) < 60.0] = 0

    cv.imshow("diff", diff)

    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    gray[np.abs(gray) < 10] = 0
    fgMask = gray

    cv.imshow("fgMask", fgMask)



    # Uklanjanje suma
    kernel = np.ones((3, 3), np.uint8)
    fgMask = cv.erode(fgMask, kernel, iterations=2)
    fgMask = cv.dilate(fgMask, kernel, iterations=2)

    cv.imshow("gray fgMask", fgMask)


    # Delimo sliku na tri dela: 25%, 50%, 25%
    height, width = fgMask.shape
    left_part = fgMask[:, :width//4]
    middle_part = fgMask[:, width//4:3*width//4]
    right_part = fgMask[:, 3*width//4:]

    small_kernel = np.ones((3, 3), np.uint8)
    large_kernel = np.ones((20, 20), np.uint8)

    left_part = cv.morphologyEx(left_part, cv.MORPH_CLOSE, small_kernel)
    middle_part = cv.morphologyEx(middle_part, cv.MORPH_CLOSE, large_kernel)
    right_part = cv.morphologyEx(right_part, cv.MORPH_CLOSE, small_kernel)

    # Spajamo nazad delove
    fgMask[:, :width//4] = left_part
    fgMask[:, width//4:3*width//4] = middle_part
    fgMask[:, 3*width//4:] = right_part

    fgMask[fgMask > 3] = 255

    cv.imshow("fgMask after morphologyEx", fgMask)


    contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        fgMask = np.zeros_like(fgMask)
        cv.drawContours(fgMask, [largest_contour], -1, (255), thickness=cv.FILLED)

    cv.imshow("Foreground Mask", fgMask)

    fgMask_inv = cv.bitwise_not(fgMask)
    fgMask_inv = cv.cvtColor(fgMask_inv, cv.COLOR_GRAY2BGR)

    fgMask = cv.cvtColor(fgMask, cv.COLOR_GRAY2BGR)

    fgImage = cv.bitwise_and(img, fgMask)
    bgImage = cv.bitwise_and(bg, fgMask_inv)

    cv.imshow('fgImage', fgImage)
    cv.imshow('bgImage', bgImage)

    # Kombinovanje foreground-a i novog backgrounda-a
    bgSub = cv.add(bgImage, fgImage)

    imgStacked = cvzone.stackImages([img, bgSub], 2, 1)
    cv.imshow('BG Subtraction', imgStacked)
    cv.imshow('Background Removed', bgSub)
    cv.imshow('Original', img)

    key = cv.waitKey(5) & 0xFF
    if ord('q') == key:
        break
    elif ord('e') == key:
        takeBgImage = 1
        print("Background Captured")
    elif ord('r') == key:
        takeBgImage = 0
        print("Ready to Capture new Background")

cv.destroyAllWindows()
video.release()
