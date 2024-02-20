
# Import module
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os

import cv2
import numpy as np

####### Homework 1 ######

def Homework1Button():
    global frameToolbox
    # Clear toolbox
    for widget in frameToolbox.winfo_children():
        widget.destroy()
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
    # Convert RGB to BGR
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
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
        directory = filedialog.askdirectory()
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
def Homework2Button():
    global frameToolbox
    global gamma
    gamma = 0.1
    global cgamma
    cgamma = 0
    # Clear toolbox
    for widget in frameToolbox.winfo_children():
        widget.destroy()
    # Add negative button
    buttonNegative=Button(frameToolbox, text="Negative", command=lambda: NegativeButton())
    buttonNegative.grid(row=0, column=0, sticky="news")
    # Add log transformation scale
    frameLog = Frame(frameToolbox, borderwidth=1, relief='solid')
    frameLog.grid(row=1, column=0, sticky="news")
    frameLog.columnconfigure(0, weight=1)
    labelLog = Label(frameLog, text="Log Transformation")
    labelLog.grid(row=0, column=0, sticky="w")
    frameCLogValues = Frame(frameLog)
    frameCLogValues.grid(row=1, column=0, sticky="news")
    labelCLog = Label(frameCLogValues, text="C")
    labelCLog.pack(side=LEFT)
    scaleCLog=Scale(frameCLogValues, from_=0, to=45, orient=HORIZONTAL, command=LogTransformScale)
    scaleCLog.pack(side=RIGHT, expand=True, fill=X)
    # Add power law (gamma) transformation
    frameGamma = Frame(frameToolbox, borderwidth=1, relief='solid')
    frameGamma.grid(row=2, column=0, sticky="news")
    frameGamma.columnconfigure(0, weight=1)
    labelGamma = Label(frameGamma, text="Power Law (Gamma) Transformation")
    labelGamma.grid(row=0, column=0, sticky="w")
    frameCGammaValues = Frame(frameGamma)
    frameCGammaValues.grid(row=1, column=0, sticky="news")
    labelCGamma = Label(frameCGammaValues, text="C")
    labelCGamma.pack(side=LEFT)
    scaleCGamma=Scale(frameCGammaValues, from_=0, to=45, orient=HORIZONTAL, command=lambda value: GammaTransformScale(value, gamma))
    scaleCGamma.pack(side=RIGHT, expand=True, fill=X)
    frameGammaGammaValues = Frame(frameGamma)
    frameGammaGammaValues.grid(row=2, column=0, sticky="news")
    labelGammaGamma = Label(frameGammaGammaValues, text="Gamma")
    labelGammaGamma.pack(side=LEFT)
    scaleGammaGamma=Scale(frameGammaGammaValues, from_=0.1, to=10, digits=3, resolution=0.1, orient=HORIZONTAL, command=lambda value: GammaTransformScale(cgamma, float(value)))
    scaleGammaGamma.pack(side=RIGHT, expand=True, fill=X)
    
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
        img_gamma = np.float16(c) * 255*(img / 255) ** gamma
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
    width = 1600
    height = 600
    root.geometry(str(width) + "x" + str(height))
     
    # Set minimum window size value
    root.minsize(width, height)
     
    # Column and row amount
    row = 4
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
    buttonOpenFolder=Button(root, text="Save Image", command=lambda: saveImage())
    buttonOpenFolder.grid(row=2, column=1, sticky="news")
    # Homework 1 button
    buttonHomework1=Button(root, text="Homework 1", command=lambda: Homework1Button())
    buttonHomework1.grid(row=3, column=0, sticky="news")
    # Homework 2 button
    buttonHomework2=Button(root, text="Homework 2", command=lambda: Homework2Button())
    buttonHomework2.grid(row=3, column=1, sticky="news")
    
    #Toolbox
    frameToolbox = Frame(root, borderwidth=1, relief='solid')
    frameToolbox.grid(row=0, column=column-1, rowspan=row, sticky="news")
    frameToolbox.columnconfigure(0, weight=1)
    
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