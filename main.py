import cv2
import numpy as np
import wx
import pyautogui

if __name__ == '__main__':

  app = wx.App(False)
  sx, sy = wx.GetDisplaySize()
  (camx, camy) = (330, 250)

  cap = cv2.VideoCapture(0)
  cap.set(3, camx)
  cap.set(4, camy)

  lowerBound = np.array([33,80,40])
  upperBound = np.array([102,255,255])
  kernelOpen = np.ones((5,5))
  kernelClose=np.ones((20,20))

  while True :
    ret, img = cap.read()

    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelOpen)

    maskFinal = maskOpen
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    print(len(conts))

    if (len(conts) == 2):
      print(2)
      x1, y1, w1, h1 = cv2.boundingRect(conts[0])
      x2, y2, w2, h2 = cv2.boundingRect(conts[1])
      cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
      cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
      cx1 = int(x1 + w1 / 2)
      cy1 = int(y1 + h1 / 2)
      cx2 =  int(x2 + w2 / 2)
      cy2 = int(y2 + h2 / 2)
      cx = int((cx1 + cx2) / 2)
      cy = int((cy1 + cy2) / 2)
      cv2.line(img, (cx1,cy1),(cx2,cy2),(255,0,0),2)
      cv2.circle(img, (cx,cy),2,(0,0,255),2)
      pyautogui.moveTo(sx-(cx*sx/camx), cy*sy/camy)

    elif (len(conts) == 1) :
      print(1)
      x, y, w, h = cv2.boundingRect(conts[0])
      cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
      cx = int(x + w / 2)
      cy = int(y + h / 2)
      cv2.circle(img, (cx,cy), int((w+h)/4), (0,0,255), 2)
      pyautogui.click(sx-(cx*sx/camx), cy*sy/camy)
      pyautogui.moveTo(sx-(cx*sx/camx), cy*sy/camy)
      
    cv2.imshow('main', img)
    if cv2.waitKey(10) == ord('q') :
      break

cap.release()
cv2.destroyAllWindows()