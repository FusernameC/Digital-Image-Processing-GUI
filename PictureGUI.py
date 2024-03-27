
# Import module
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import ctypes

import cv2
import numpy as np

####### Homework 1 ######

def Homework1Button():
    global frameToolboxOuter
    global frameToolbox
    
    # Clear toolbox
    for widget in frameToolboxOuter.winfo_children():
        widget.destroy()
        
    frameToolbox = Frame(frameToolboxOuter)
    frameToolbox.columnconfigure(0, weight=1)
    frameToolboxOuter.create_window((0, 0), window=frameToolbox, anchor='nw', tags='frame')
    
    def onFrameConfigure(frameToolboxOuter):
        '''Reset the scroll region to encompass the inner frame'''
        frameToolboxOuter.configure(scrollregion=frameToolboxOuter.bbox("all"))
        frameToolboxOuter.itemconfig('frame', width=frameToolboxOuter.winfo_width()-5)
    frameToolbox.bind("<Configure>", lambda event, frameToolboxOuter=frameToolboxOuter: onFrameConfigure(frameToolboxOuter))
    
    
    # Add red layer button
    buttonRedLayer=Button(frameToolbox, text="Red Layer", command=lambda: RGBButton("r"))
    buttonRedLayer.grid(row=0, column=0, sticky="ew")
    # Add green layer button
    buttonGreenLayer=Button(frameToolbox, text="Green Layer", command=lambda: RGBButton("g"))
    buttonGreenLayer.grid(row=1, column=0, sticky="ew")
    # Add blue layer button
    buttonBlueLayer=Button(frameToolbox, text="Blue Layer", command=lambda: RGBButton("b"))
    buttonBlueLayer.grid(row=2, column=0, sticky="ew")
    # Add grayscale button
    buttonGrayscale=Button(frameToolbox, text="Grayscale", command=lambda: GrayButton())
    buttonGrayscale.grid(row=3, column=0, sticky="ew")
    # Add rotate and resize button
    buttonRotateAndResize=Button(frameToolbox, text="Rotate and resize", command=lambda: RotateAndResizeButton(15, 0.9))
    buttonRotateAndResize.grid(row=4, column=0, sticky="ew")
    # Add crop button
    buttonCrop=Button(frameToolbox, text="Crop", command=lambda: CropButton((0.25, 0.25)))
    buttonCrop.grid(row=5, column=0, sticky="news")

def convertRawToTk(rawImage):
    image = ImageTk.PhotoImage(resizeImagePIL(rawImage))
    return image

def convertPILToCV2(pil_image):
    open_cv_image = np.array(pil_image)
    if open_cv_image.dtype != bool:
        # Convert RGB to BGR
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image

def resizeImagePIL(image):
    w, h = image.size
    if (w/labelImage.winfo_width() > h/labelImage.winfo_height()):
        image = image.resize((labelImage.winfo_width(), int(h/(w/labelImage.winfo_width()))))
    elif (w/labelImage.winfo_width() < h/labelImage.winfo_height()):
        image = image.resize((int(w/(h/labelImage.winfo_height())), labelImage.winfo_height()))
    else:
        image = image.resize((labelImage.winfo_width(), labelImage.winfo_height()))
    return image

def resizeImageCV2(image):
    h, w, _ = image.shape
    if (w/labelImage.winfo_width() > h/labelImage.winfo_height()):
        image = cv2.resize(image, (labelImage.winfo_width(), int(h/(w/labelImage.winfo_width()))))
    elif (w/labelImage.winfo_width() < h/labelImage.winfo_height()):
        image = cv2.resize(image, (int(w/(h/labelImage.winfo_height())), labelImage.winfo_height()))
    else:
        image = cv2.resize(image, (labelImage.winfo_width(), labelImage.winfo_height()))
    return image

# Resize elements when resize window
def refreshElements(event):
    global labelName
    if itemImgEdit != None:
        image = convertRawToTk(itemImgEdit)
        labelImage.configure(image=image)
        labelImage.image=image

def clickFile(label):
    global directory
    global name
    global labelImage
    global breakLoop
    global itemImg
    global itemImgEdit
    # Get file name
    name=str(label.cget('text'))
    # Get file path
    path=directory+"/"+name
    # Get image
    rawImage = Image.open(path)
    # Save file image
    itemImg = rawImage
    itemImgEdit = rawImage
    # Convert image
    image = convertRawToTk(rawImage)
    # Show image
    labelImage.configure(image=image)
    labelImage.image=image

    
    # Break out of rotate and resize loop
    breakLoop = True
	
def openFolder():
    global directory
    global frameItems
    global scrollbar
    row = 0
    col = 0
    formats=[".png", ".jpg", ".jpeg", ".bmp"]
    tempDir = filedialog.askdirectory()
    if tempDir != "":
        directory = tempDir
        fileList = os.listdir(directory)
        # Clear frame
        for widget in frameItems.winfo_children():
            widget.destroy()
        tempFrame = Frame(frameItems)
        frameItems.create_window((0, 0), window=tempFrame, anchor='nw')
        
        def onFrameConfigure(frameItems):
            '''Reset the scroll region to encompass the inner frame'''
            frameItems.configure(scrollregion=frameItems.bbox("all"))
        tempFrame.bind("<Configure>", lambda event, frameItems=frameItems: onFrameConfigure(frameItems))
        
        # Display folder items
        for format in formats:
            for name in fileList:
                if name.endswith(format):
                    path = directory+"/"+name
                    thumb = Image.open(path)
                    thumb.thumbnail((80, 80))
                    image = ImageTk.PhotoImage(thumb)
                    label = Label(tempFrame, image=image, text=name, compound='top')
                    label.image = image
                    label.grid(row=row, column=col)
                    label.bind("<Button-1>",lambda e, label=label:clickFile(label))
                    if col < 2:
                        col+=1
                    else:
                        col=0
                        row+=1

def saveImage():
    if itemImgEdit != None:
        itemImgEdit.save(filedialog.asksaveasfilename(defaultextension=".png", filetypes=(("PNG", ".png"), ("Bitmap", ".bmp"), ("JPEG", ("*.jpg", "*.jpeg", "*.jpe", "*.jfif")))))

def RGBButton(type):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Split image into blue, green and red layer
        b,g,r = cv2.split(img)
        # Red layer
        if (type == "r"):
            imgConverted = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        # Green layer
        elif (type == "g"):
            imgConverted = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        # Blue layer
        elif (type == "b"):
            imgConverted = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil

def GrayButton():
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        img = convertPILToCV2(itemImg)
        # Grayscale and convert to RGB
        grayImg = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
        # Save image
        itemImgEdit = Image.fromarray(grayImg)
        # Convert to Pillow image
        grayImgPil = convertRawToTk(Image.fromarray(grayImg))
        # Show Image
        labelImage.configure(image=grayImgPil)
        labelImage.image=grayImgPil
    
def RotateAndResizeButton(angle, zoom_factor):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = False
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Convert image from BGR to RGB 
        imgConverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        labelImage.update_idletasks()
        # Get image height and width
        (h, w) = img.shape[:2]
        # Calculate the center of the image
        center = (w // 2, h // 2)
        def rotAndRes(i=0):
            if  breakLoop == True or i+1 == 100: return
            global root
            global itemImgEdit
            # Get the rotation matrix
            M = cv2.getRotationMatrix2D(center, angle*i, pow(zoom_factor, i))
            # Rotate the image and zoom out
            imgEdited = cv2.warpAffine(img, M, (w, h))
            # Convert image from BGR to RGB
            imgConverted = cv2.cvtColor(imgEdited, cv2.COLOR_BGR2RGB)
            # Save image
            itemImgEdit = Image.fromarray(imgConverted)
            # Convert to Pillow image
            imgPil = convertRawToTk(Image.fromarray(imgConverted))
            # Show Image
            labelImage.configure(image=imgPil)
            labelImage.image=imgPil
            labelImage.update_idletasks()
            # Delay by 1s
            root.after(1000, rotAndRes, i+1)
        rotAndRes()
        
        
def CropButton(crop_size):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        img = resizeImageCV2(img)
        # Get image height and width
        (h, w) = img.shape[:2]
        # Calculate the center of the image
        center = (w // 2, h // 2)
        # Calculate the crop size
        crop_w, crop_h = crop_size
        # Crop the image
        crop = img[int(center[1] - crop_h * h) : int(center[1] + crop_h * h), int(center[0] - crop_w * w) : int(center[0] + crop_w * w)]
        imgConverted = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = ImageTk.PhotoImage(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
    
####### Homework 2 ######
## Visual
def Homework2Button():
    global gamma, cgamma
    gamma , cgamma = 0.1, 0 
    global r1PL, s1PL, r2PL, s2PL
    r1PL, s1PL, r2PL, s2PL = 0, 0, 0, 0
    global kGauss, sGauss
    kGauss, sGauss = 1, 0
    global kUM, sUM
    kUM, sUM = 1, 0
    global knHB, sHB, kHB
    knHB, sHB, kHB = 1, 0, 1
    
    global frameToolboxOuter
    global frameToolbox
    
    # Clear toolbox
    for widget in frameToolboxOuter.winfo_children():
        widget.destroy()
        
    frameToolbox = Frame(frameToolboxOuter)
    frameToolbox.columnconfigure(0, weight=1)
    frameToolboxOuter.create_window((0, 0), window=frameToolbox, anchor='nw', tags='frame')
    
    def onFrameConfigure(frameToolboxOuter):
        '''Reset the scroll region to encompass the inner frame'''
        frameToolboxOuter.configure(scrollregion=frameToolboxOuter.bbox("all"))
        frameToolboxOuter.itemconfig('frame', width=frameToolboxOuter.winfo_width()-5)
    frameToolbox.bind("<Configure>", lambda event, frameToolboxOuter=frameToolboxOuter: onFrameConfigure(frameToolboxOuter))
    # Add negative button
    buttonNegative=Button(frameToolbox, text="Negative", command=lambda: NegativeButton())
    buttonNegative.grid(row=0, column=0, sticky="news")
    # Add log transformation scale
    createSliderElements("Log Transformation", ["C"], LogTransformScale, [0], [[1, 45, 1, 1]], 1)
    # Add power law (gamma) transformation
    createSliderElements("Power Law (Gamma) Transformation", ["C", "Gamma"], GammaTransformScale, [cgamma, gamma], [[1, 45, 1, 1], [0.1, 10, 3, 0.1]], 2)
    # Add piecewise-linear transformation
    createSliderElements("Piecewise-Linear Transformation", ["R1", "S1", "R2", "S2"], piecewiseLinearTransform, [r1PL, s1PL, r2PL, s2PL], [[0, 255, 1, 1], [0, 255, 1, 1], [0, 255, 1, 1], [0, 255, 1, 1]], 3)
    # Add histogram equalization
    buttonHistogramEqualization=Button(frameToolbox, text="Histogram Equalization", command=lambda: histogramEqualization())
    buttonHistogramEqualization.grid(row=4, column=0, sticky="news")
    # Add box filter
    createSliderElements("Smoothing(Box Filter)", ["Kernel size"], boxFilter, [0], [[1, 50, 1, 1]], 6)
    # Add Gaussian filter
    createSliderElements("Smoothing(Gaussian Filter)", ["Kernel size", "Sigma"], gaussianFilter, [kGauss, sGauss], [[1, 50, 1, 2], [0, 10, 1, 1]], 7)
    # Add median filter
    createSliderElements("Smoothing (Median Filter)", ["Kernel size"], medianFilter, [0], [[1, 50, 1, 1]], 8)
    # Add min filter
    createSliderElements("Smoothing (Min Filter)", ["Kernel size"], minFilter, [0], [[1, 50, 1, 1]], 9)
    # Add max filter
    createSliderElements("Smoothing (Max Filter)", ["Kernel size"], maxFilter, [0], [[1, 50, 1, 1]], 10)
    # Add midpoint filter
    createSliderElements("Smoothing (Midpoint Filter)", ["Kernel size"], midpointFilter, [0], [[1, 50, 1, 1]], 11)
    # Add Laplacian filter
    createSliderElements("Sharpening (Laplacian Filter)", ["Kernel size"], laplacianFilter, [0], [[1, 10, 1, 1]], 12)
    # Add unsharp masking filter
    createSliderElements("Smoothing (Unsharp Masking Filter)", ["Kernel size", "Sigma"], unsharpMasking, [kUM, sUM], [[1, 50, 1, 2], [0, 10, 1, 1]], 13)
    # Add high boost filter
    createSliderElements("Smoothing (High Boost Filter)", ["Kernel size", "Sigma", "K"], highBoostFilter, [knHB, sHB, kHB], [[1, 50, 1, 2], [0, 10, 1, 1], [1, 10, 1, 1]], 14)
    
def createSliderElements(name, sliderNames, function, args, scaleValues, row):
    def replace(l, e, i):
        tempList = l
        tempList[i] = e
        return tempList
    global frameToolbox
    frameMain = Frame(frameToolbox, borderwidth=1, relief='solid')
    frameMain.grid(row=row, column=0, sticky="news")
    frameMain.columnconfigure(0, weight=1)
    labelMain = Label(frameMain, text=name)
    labelMain.grid(row=0, column=0, sticky="w")
    i = 0
    for sliderName in sliderNames:
        frameSlider = Frame(frameMain)
        frameSlider.grid(row=i+1, column=0, sticky="news")
        label = Label(frameSlider, text=sliderName)
        label.pack(side=LEFT)
        scale = Scale(frameSlider, from_=scaleValues[i][0], to=scaleValues[i][1], digits=scaleValues[i][2], resolution=scaleValues[i][3], orient=HORIZONTAL, command=lambda value, i = i: function(*replace(args, value, i)))
        scale.pack(side=RIGHT, expand=True, fill=X)
        i+=1
    
## Algorithm
def NegativeButton():
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Transformation
        img_neg = 255 - img
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(img_neg, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def LogTransformScale(c):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Transformation
        img_log = np.float16(c) * (np.log(1 + img))
        img_log = np.array(img_log, dtype = np.uint8) 
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(img_log, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
    
def GammaTransformScale(c, g):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    global cgamma
    global gamma
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Transformation
        img_gamma = np.float16(c) * 255*(img / 255) ** float(gamma)
        img_gamma = np.array(img_gamma, dtype = np.uint8)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        cgamma = c
        gamma = g
        
def piecewiseLinearTransform(r1, s1, r2, s2):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    global r1PL, s1PL, r2PL, s2PL
    r1 = int(r1)
    s1 = int(s1)
    r2 = int(r2)
    s2 = int(s2)
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Convert image to grayscale
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
        # Perform contrast stretching
        imgConverted = img
        imgConverted[np.logical_and(imgConverted >= 0, imgConverted <= r1)] = (s1/r1 * imgConverted[np.logical_and(imgConverted >= 0, imgConverted <= r1)]) if (abs(r1) > 0) else (s1)
        imgConverted[np.logical_and(imgConverted > r1, imgConverted <= r2)] = (s2 - s1)/(r2 - r1) * (imgConverted[np.logical_and(imgConverted > r1, imgConverted <= r2)] - r1) + s1
        imgConverted[np.logical_and(imgConverted > r2, imgConverted <= 255)] = ((255 - s2)/(255 - r2)) * (imgConverted[np.logical_and(imgConverted > r2, imgConverted <= 255)] - r2) + s2
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        r1PL, s1PL, r2PL, s2PL = r1, s1, r2, s2
        
def histogramEqualization():
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Convert image to grayscale
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
        #Perform histogram equalization
        imgConverted = cv2.equalizeHist(img)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def boxFilter(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform box filtering
        imgConverted = cv2.boxFilter(img, ddepth=-1, ksize=(int(ksize), int(ksize)))
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def gaussianFilter(ksize, sigma):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    global kGauss, sGauss
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform gaussian filtering
        imgConverted = cv2.GaussianBlur(img, ksize=(int(ksize), int(ksize)), sigmaX=int(sigma))
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        kGauss, sGauss = ksize, sigma

def medianFilter(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform median filtering
        imgConverted = cv2.medianBlur(img, ksize=int(ksize))
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
    
def minFilter(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform min filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ksize), int(ksize)))
        imgConverted = cv2.erode(img, kernel=kernel)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def maxFilter(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform max filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ksize), int(ksize)))
        imgConverted = cv2.dilate(img, kernel=kernel)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
    
def midpointFilter(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform midpoint filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ksize), int(ksize)))
        imgMin = cv2.erode(img, kernel=kernel)
        imgMax = cv2.dilate(img, kernel=kernel)
        imgConverted = cv2.addWeighted(imgMin, 0.5, imgMax, 0.5, 0)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def laplacianFilter(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform laplacian filtering
        laplacian = cv2.Laplacian(img, ddepth=-1, ksize=int(ksize))
        imgConverted = cv2.subtract(img, laplacian)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
    
def unsharpMasking(ksize, sigma):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    global kUM, sUM
    breakLoop = True
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform unsharp mask filtering
        gaussian = cv2.GaussianBlur(img, ksize=(int(ksize), int(ksize)), sigmaX=int(sigma))
        mask = cv2.subtract(img, gaussian)
        imgConverted = cv2.add(img, mask)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        kUM, sUM = ksize, sigma
        
def highBoostFilter(ksize, sigma, k):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    global knHB, sHB, kHB
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Perform high boost filtering
        gaussian = cv2.GaussianBlur(img, ksize=(int(ksize), int(ksize)), sigmaX=int(sigma))
        mask = cv2.subtract(img, gaussian)
        imgConverted = img
        for i in range(int(k)):
            imgConverted = cv2.add(imgConverted, mask)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        knHB, sHB, kHB = ksize, sigma, k

####### Classwork 2 ######
def Classwork2Button():
    global frameToolboxOuter
    global frameToolbox
    global nLPB, D0LPB
    nLPB, D0LPB = 1, 1
    global nHPB, D0HPB
    nHPB, D0HPB = 1, 1
    
    # Clear toolbox
    for widget in frameToolboxOuter.winfo_children():
        widget.destroy()
        
    frameToolbox = Frame(frameToolboxOuter)
    frameToolbox.columnconfigure(0, weight=1)
    frameToolboxOuter.create_window((0, 0), window=frameToolbox, anchor='nw', tags='frame')
    
    def onFrameConfigure(frameToolboxOuter):
        '''Reset the scroll region to encompass the inner frame'''
        frameToolboxOuter.configure(scrollregion=frameToolboxOuter.bbox("all"))
        frameToolboxOuter.itemconfig('frame', width=frameToolboxOuter.winfo_width()-5)
    frameToolbox.bind("<Configure>", lambda event, frameToolboxOuter=frameToolboxOuter: onFrameConfigure(frameToolboxOuter))
    
    # Add lowpass ideal filter 
    createSliderElements("Lowpass Ideal Filter", ["D0"], lowpassIdealFilter, [0], [[1, 200, 1, 1]], 0)
    # Add lowpass butterworth filter
    createSliderElements("Lowpass Butterworth Filter", ["n", "D0"], lowpassButterworthFilter, [nLPB, D0LPB], [[1, 50, 1, 1], [1, 200, 1, 1]], 1)
    # Add lowpass Gaussian filter 
    createSliderElements("Lowpass Gaussian Filter", ["D0"], lowpassGaussianFilter, [0], [[1, 200, 1, 1]], 2)
    # Add highpass ideal filter
    createSliderElements("Highpass Ideal Filter", ["D0"], highpassIdealFilter, [0], [[1, 200, 1, 1]], 3)
    #Add highpass butterworth filter
    createSliderElements("Highpass Butterworth Filter", ["n", "D0"], highpassButterworthFilter, [nHPB, D0HPB], [[1, 50, 1, 1], [1, 200, 1, 1]], 4)
    # Add lowpass Gaussian filter 
    createSliderElements("Highpass Gaussian Filter", ["D0"], highpassGaussianFilter, [0], [[1, 200, 1, 1]], 5)
    # Add apply button
    buttonApply=Button(frameToolbox, text="Apply", command=lambda: applyButton())
    buttonApply.grid(row=10, column=0, sticky="ews")

def lowpassIdealFilter(D0):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Divide into 3 channels
        channels = cv2.split(img)
        newChannels = []
        # Perform filter on each channel
        for channel in channels:
            # Perform filter
            F = np.fft.fft2(channel)
            F = np.fft.fftshift(F)
            M, N = np.array(channel).shape
            
            u = np.arange(0, M)-M/2
            v = np.arange(0, N)-N/2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt(np.power(U, 2) + np.power(V, 2))
            H = np.array(D<=int(D0), 'float')
            G = H*F
            
            G = np.fft.ifftshift(G)
            newChannels.append(np.real(np.fft.ifft2(G)))
        # Merge channels
        imgConverted = cv2.merge(newChannels).astype(np.uint8)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def lowpassButterworthFilter(n, D0):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    global nLP, D0LP
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Divide into 3 channels
        channels = cv2.split(img)
        newChannels = []
        # Perform filter on each channel
        for channel in channels:
            # Perform filter
            F = np.fft.fft2(channel)
            F = np.fft.fftshift(F)
            M, N = channel.shape
            
            u = np.arange(0, M)-M/2
            v = np.arange(0, N)-N/2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt(np.power(U, 2) + np.power(V, 2))
            H = 1/np.power(1 + (D/int(D0)), (2*int(n)))
            G = H*F
            
            G = np.fft.ifftshift(G)
            newChannels.append(np.real(np.fft.ifft2(G)))
        # Merge channels
        imgConverted = cv2.merge(newChannels).astype(np.uint8)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        nLPB, D0LPB = n, D0
        
def lowpassGaussianFilter(D0):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    global nLP, D0LP
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        # Divide into 3 channels
        channels = cv2.split(img)
        newChannels = []
        # Perform filter on each channel
        for channel in channels:
            # Perform filter
            F = np.fft.fft2(channel)
            F = np.fft.fftshift(F)
            M, N = channel.shape
            
            u = np.arange(0, M)-M/2
            v = np.arange(0, N)-N/2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt(np.power(U, 2) + np.power(V, 2))
            H = np.power(np.e, (-np.power(D, 2)/(2*np.power(int(D0), 2))))
            G = H*F
            
            G = np.fft.ifftshift(G)
            newChannels.append(np.real(np.fft.ifft2(G)))
        # Merge channels
        imgConverted = cv2.merge(newChannels).astype(np.uint8)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil

def highpassIdealFilter(D0):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    if itemImgEdit != None:
        # Get image
        img = cv2.cvtColor(convertPILToCV2(itemImg), cv2.COLOR_BGR2GRAY)
        # Perform filter
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N = img.shape
        
        u = np.arange(0, M)-M/2
        v = np.arange(0, N)-N/2
        idx = (u > M/2)
        u[idx] = u[idx] - M
        idy = (v > N/2)
        v[idy] = v[idy] - N
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = np.array(D>int(D0), 'float')
        G = H*F
        
        G = np.fft.ifftshift(G)
        imgConverted = np.real(np.fft.ifft2(G))
        imgConverted -= np.amin(np.real(np.fft.ifft2(G)))
        imgConverted *= 255.0/np.amax(imgConverted)
        
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil

def highpassButterworthFilter(n, D0):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    global nLP, D0LP
    if itemImgEdit != None:
        # Get image
        img = cv2.cvtColor(convertPILToCV2(itemImg), cv2.COLOR_BGR2GRAY)
        # Perform filter
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N = img.shape
        
        u = np.arange(0, M)-M/2
        v = np.arange(0, N)-N/2
        idx = (u > M/2)
        u[idx] = u[idx] - M
        idy = (v > N/2)
        v[idy] = v[idy] - N
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = 1/np.power(1 + (int(D0)/(D)), (2*int(n)))
        G = H*F
        
        G = np.fft.ifftshift(G)
        imgConverted = np.real(np.fft.ifft2(G))
        imgConverted -= np.amin(np.real(np.fft.ifft2(G)))
        imgConverted *= 255.0/np.amax(imgConverted)
        
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        nHPB, D0HPB = n, D0
        
def highpassGaussianFilter(D0):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    if itemImgEdit != None:
        # Get image
        img = cv2.cvtColor(convertPILToCV2(itemImg), cv2.COLOR_BGR2GRAY)
        # Perform filter
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N = img.shape
        
        u = np.arange(0, M)-M/2
        v = np.arange(0, N)-N/2
        idx = (u > M/2)
        u[idx] = u[idx] - M
        idy = (v > N/2)
        v[idy] = v[idy] - N
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = 1 - np.power(np.e, (-np.power(D, 2)/(2*np.power(int(D0), 2))))
        G = H*F
        
        G = np.fft.ifftshift(G)
        imgConverted = np.real(np.fft.ifft2(G))
        imgConverted -= np.amin(np.real(np.fft.ifft2(G)))
        imgConverted *= 255.0/np.amax(imgConverted)
        
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil

####### Classwork 3 ######
def Classwork3Button():
    global frameToolboxOuter
    global frameToolbox
    
    # Clear toolbox
    for widget in frameToolboxOuter.winfo_children():
        widget.destroy()
        
    frameToolbox = Frame(frameToolboxOuter)
    frameToolbox.columnconfigure(0, weight=1)
    frameToolboxOuter.create_window((0, 0), window=frameToolbox, anchor='nw', tags='frame')
    
    def onFrameConfigure(frameToolboxOuter):
        '''Reset the scroll region to encompass the inner frame'''
        frameToolboxOuter.configure(scrollregion=frameToolboxOuter.bbox("all"))
        frameToolboxOuter.itemconfig('frame', width=frameToolboxOuter.winfo_width()-5)
    frameToolbox.bind("<Configure>", lambda event, frameToolboxOuter=frameToolboxOuter: onFrameConfigure(frameToolboxOuter))

    # Add erosion
    createSliderElements("Erosion", ["Kernel size"], erosion, [0], [[1, 200, 1, 1]], 0)
    # Add dilation
    createSliderElements("Dilation", ["Kernel size"], dilation, [0], [[1, 200, 1, 1]], 1)
    # Add opening
    createSliderElements("Opening", ["Kernel size"], opening, [0], [[1, 200, 1, 1]], 2)
    # Add closing
    createSliderElements("Closing", ["Kernel size"], closing, [0], [[1, 200, 1, 1]], 3)
    # Add boundary extraction
    createSliderElements("Boundary extraction", ["Kernel size"], boundaryExtraction, [0], [[1, 200, 1, 1]], 4)
    # Add region filling
    buttonRF=Button(frameToolbox, text="Region filling", command=lambda: regionFilling())
    buttonRF.grid(row=5, column=0, sticky="ews")
    # Add apply button
    buttonApply=Button(frameToolbox, text="Apply", command=lambda: applyButton())
    buttonApply.grid(row=10, column=0, sticky="ews")
    
    
def erosion(ksize=1, kernel=[]):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    ksize = int(ksize)
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        if img.dtype != bool:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        else:
            img = img.astype(np.uint8) * 255
        # Perform erosion
        if (len(kernel) == 0):
            kernel = np.ones((ksize, ksize), np.uint8)
        imgConverted = cv2.erode(img, kernel)
        # Convert from BRG to RGB
        # imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def dilation(ksize=1, kernel=[]):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    ksize = int(ksize)
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        if img.dtype != bool:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        else:
            img = img.astype(np.uint8) * 255
        # Perform dilation
        if (len(kernel) == 0):
            kernel = np.ones((ksize, ksize), np.uint8)
        imgConverted = cv2.dilate(img, kernel)
        # Convert from BRG to RGB
        # imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def opening(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    ksize = int(ksize)
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        if img.dtype != bool:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        else:
            img = img.astype(np.uint8) * 255
        # Perform opening
        kernel = np.ones((ksize, ksize), np.uint8)
        imgConverted = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # Convert from BRG to RGB
        imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def closing(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    ksize = int(ksize)
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        if img.dtype != bool:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        else:
            img = img.astype(np.uint8) * 255
        # Perform closing
        kernel = np.ones((ksize, ksize), np.uint8)
        imgConverted = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Convert from BRG to RGB
        # imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil
        
def boundaryExtraction(ksize):
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    ksize = int(ksize)
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        if img.dtype != bool:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        else:
            img = img.astype(np.uint8) * 255
        # Perform boundary extraction
        kernel = np.ones((ksize, ksize), np.uint8)
        eroded = cv2.erode(img, kernel)
        imgConverted = cv2.subtract(img, eroded)
        # Convert from BRG to RGB
        # imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil     

def regionFilling():
    global labelImage
    global itemImg
    global itemImgEdit
    global breakLoop
    if itemImgEdit != None:
        # Get image
        img = convertPILToCV2(itemImg)
        if img.dtype != bool:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        else:
            img = img.astype(np.uint8) * 255
        # Perform region filling
        imgConverted = img.copy()
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print("aaaaaaaaaaaaa" + str(len(contours)))
        if len(contours) > 0:
            cv2.drawContours(imgConverted, contours, -1, (255,), cv2.FILLED)
        # Convert from BRG to RGB
        # imgConverted = cv2.cvtColor(imgConverted, cv2.COLOR_BGR2RGB)
        # Save image
        itemImgEdit = Image.fromarray(imgConverted)
        # Convert to Pillow image
        imgPil = convertRawToTk(Image.fromarray(imgConverted))
        # Show Image
        labelImage.configure(image=imgPil)
        labelImage.image=imgPil

def applyButton():
    global itemImg
    global itemImgEdit
    itemImg = itemImgEdit
    
####### Homework 3 ######
def Homework3Button():
    global frameToolboxOuter
    global frameToolbox
    
    # Clear toolbox
    for widget in frameToolboxOuter.winfo_children():
        widget.destroy()
        
    frameToolbox = Frame(frameToolboxOuter)
    frameToolbox.columnconfigure(0, weight=1)
    frameToolboxOuter.create_window((0, 0), window=frameToolbox, anchor='nw', tags='frame')
    
    def onFrameConfigure(frameToolboxOuter):
        '''Reset the scroll region to encompass the inner frame'''
        frameToolboxOuter.configure(scrollregion=frameToolboxOuter.bbox("all"))
        frameToolboxOuter.itemconfig('frame', width=frameToolboxOuter.winfo_width()-5)
    frameToolbox.bind("<Configure>", lambda event, frameToolboxOuter=frameToolboxOuter: onFrameConfigure(frameToolboxOuter))
    
    # Add HW3-1 Button
    buttonHW31=Button(frameToolbox, text="HW3-1", command=lambda: HW31())
    buttonHW31.grid(row=0, column=0, sticky="ews")
    # Add HW3-2 1 pass Button
    buttonHW32_1=Button(frameToolbox, text="HW3-2: 1 pass", command=lambda: HW32_1pass())
    buttonHW32_1.grid(row=1, column=0, sticky="ews")
    # Add HW3-2 10 pass Button
    buttonHW32_10=Button(frameToolbox, text="HW3-2: 10 pass", command=lambda: HW32_10pass())
    buttonHW32_10.grid(row=2, column=0, sticky="ews")
    # Add HW3-2 100 pass Button
    buttonHW32_100=Button(frameToolbox, text="HW3-2: 100 pass", command=lambda: HW32_100pass())
    buttonHW32_100.grid(row=3, column=0, sticky="ews")
    
def HW31():
    global itemImg
    itemSave = itemImg
    lowpassGaussianFilter(25)
    applyButton()
    highpassGaussianFilter(25)
    itemImg = itemSave
    
def HW32_1pass():
    highpassGaussianFilter(30)
    
def HW32_10pass():
    global itemImg
    itemSave = itemImg
    for i in range(10):
        highpassGaussianFilter(30)
        applyButton()
    itemImg = itemSave
        
    
def HW32_100pass():
    global itemImg
    itemSave = itemImg
    for i in range(100):
        highpassGaussianFilter(30)
        applyButton()
    itemImg = itemSave
    
####### Homework 4 ######
def Homework4Button():
    global frameToolboxOuter
    global frameToolbox
    
    # Clear toolbox
    for widget in frameToolboxOuter.winfo_children():
        widget.destroy()
        
    frameToolbox = Frame(frameToolboxOuter)
    frameToolbox.columnconfigure(0, weight=1)
    frameToolboxOuter.create_window((0, 0), window=frameToolbox, anchor='nw', tags='frame')
    
    def onFrameConfigure(frameToolboxOuter):
        '''Reset the scroll region to encompass the inner frame'''
        frameToolboxOuter.configure(scrollregion=frameToolboxOuter.bbox("all"))
        frameToolboxOuter.itemconfig('frame', width=frameToolboxOuter.winfo_width()-5)
    frameToolbox.bind("<Configure>", lambda event, frameToolboxOuter=frameToolboxOuter: onFrameConfigure(frameToolboxOuter))
    
    kernel = np.zeros((3, 3), np.uint8)
    for i in range(3):
        kernel[i][i] = 1
    # Add Erode button
    buttonErode=Button(frameToolbox, text="Erosion", command=lambda: erosion(kernel=kernel))
    buttonErode.grid(row=0, column=0, sticky="ews")
    # Add Dilate Button
    buttonDilate=Button(frameToolbox, text="Dilation", command=lambda: dilation(kernel=kernel))
    buttonDilate.grid(row=1, column=0, sticky="ews")
    # Add boundary extraction
    createSliderElements("Boundary extraction", ["Kernel size"], boundaryExtraction, [0], [[1, 200, 1, 1]], 2)
    # Add apply button
    buttonApply=Button(frameToolbox, text="Apply", command=lambda: applyButton())
    buttonApply.grid(row=10, column=0, sticky="ews")

####### Main ######
if __name__ == "__main__":
    # Variable for storing file path
    directory=""
    name = ""
    itemImg = None
    itemImgEdit = None
    
    
    # Create object
    root = Tk()
    root.title('Image viewer')
    
    # Adjust size
    width = int((1600)/1920*root.winfo_screenwidth())
    height = int((600)/1080*root.winfo_screenheight())
    root.geometry(str(width) + "x" + str(height))
     
    # Set minimum window size value
    root.minsize(width, height)
    root.state('zoomed')
     
    # Column and row amount
    row = 6
    column = 4
    
    # Resize columns and rows
    root.columnconfigure(2, weight=2)
    root.columnconfigure(3, minsize=250)
    root.rowconfigure(0, weight=3)
    
    # Resize image when resize window
    root.bind("<Configure>", refreshElements)
    
    # Image label
    labelImage=Label(borderwidth=1, relief='solid')
    labelImage.grid(row=0, column=2, rowspan=row, sticky="news")
    
    # Folder and scrollbar
    frameFolder=Frame(root, borderwidth=1, relief='solid')
    frameFolder.grid(row=0, column=0, columnspan=column-2, sticky="news")
    frameItems=Canvas(frameFolder)
    frameItems.pack(side='left', fill='both')
    
    scrollbar=Scrollbar(frameFolder)
    scrollbar.pack(side='right', fill='both')

    frameItems.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=frameItems.yview)
    frameItems.config(scrollregion=frameItems.bbox("all"))
    
    # Main Buttons
    # Open folder button
    buttonOpenFolder=Button(root, text="Open Folder", command=lambda: openFolder())
    buttonOpenFolder.grid(row=2, column=0, sticky="news")
    # Save image button
    buttonSaveImage=Button(root, text="Save Image", command=lambda: saveImage())
    buttonSaveImage.grid(row=2, column=1, sticky="news")
    # Homework 1 button
    buttonHomework1=Button(root, text="Homework 1", command=lambda: Homework1Button())
    buttonHomework1.grid(row=3, column=0, sticky="news")
    # Homework 2 button
    buttonHomework2=Button(root, text="Homework 2", command=lambda: Homework2Button())
    buttonHomework2.grid(row=3, column=1, sticky="news")
    # Homework 3 button
    buttonHomework3=Button(root, text="Homework 3", command=lambda: Homework3Button())
    buttonHomework3.grid(row=4, column=0, sticky="news")
    # Homework 4 button
    buttonHomework4=Button(root, text="Homework 4", command=lambda: Homework4Button())
    buttonHomework4.grid(row=4, column=1, sticky="news")
    # Class work 2 button
    buttonClasswork2=Button(root, text="Classwork 2", command=lambda: Classwork2Button())
    buttonClasswork2.grid(row=5, column=0, sticky="news")
    # Class work 3 button
    buttonClasswork2=Button(root, text="Classwork 3", command=lambda: Classwork3Button())
    buttonClasswork2.grid(row=5, column=1, sticky="news")
    
    #Toolbox
    frameToolboxContainer= Frame(root, borderwidth=1, relief='solid')
    frameToolboxContainer.grid(row=0, column=column-1, rowspan=row, sticky="news")
    frameToolboxOuter = Canvas(frameToolboxContainer)
    frameToolboxOuter.columnconfigure(0, weight=1)
    frameToolboxOuter.pack(side='left', fill='both', expand=True)
    
    scrollbarToolbox=Scrollbar(frameToolboxContainer)
    scrollbarToolbox.pack(side='right', fill='both')

    frameToolboxOuter.config(yscrollcommand=scrollbarToolbox.set)
    scrollbarToolbox.config(command=frameToolboxOuter.yview)
    frameToolboxOuter.config(scrollregion=frameToolboxOuter.bbox("all"))
    
    
    # Menu
    # menubar = Menu(root)
    # HW1menu = Menu(menubar, tearoff=0)
    # HW1menu.add_command(label="Red Layer", command=lambda: RGBButton("r"))
    # HW1menu.add_command(label="Green Layer", command=lambda: RGBButton("g"))
    # HW1menu.add_command(label="Blue Layer", command=lambda: RGBButton("b"))
    # HW1menu.add_command(label="Grayscale", command=lambda: GrayButton())
    # HW1menu.add_command(label="Rotate And Resize", command=lambda: RotateAndResizeButton(15, 0.9))
    # HW1menu.add_command(label="Crop", command=lambda: CropButton((0.25, 0.25)))
    # menubar.add_cascade(label="HW1", menu=HW1menu)
    # root.config(menu=menubar)
    
    # Execute tkinter
    root.mainloop()