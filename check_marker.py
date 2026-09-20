import cv2

im = cv2.imread("aruco_marker.png", 0)
print("image size:", im.shape)                      # expect (300, 300)
im = cv2.copyMakeBorder(im, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)  # add quiet zone
for name in dir(cv2.aruco):
    if not name.startswith("DICT_"):
        continue
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    if hasattr(cv2.aruco, "ArucoDetector"):
        _, ids, _ = cv2.aruco.ArucoDetector(dic).detectMarkers(im)
    else:
        _, ids, _ = cv2.aruco.detectMarkers(im, dic)
    if ids is not None:
        print(name, ids.ravel().tolist())