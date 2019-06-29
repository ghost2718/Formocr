import cv2
import numpy as np
import os
import subprocess
import yaml
import torch
from models.model import Classifier
from torchvision import transforms
from PIL import Image
import PIL
from torch.autograd import Variable
import itertools
import shutil
import csv

class FormOcr():
    """This is the FormOcr Class it encapsulate all the logic and methods required to process the Image and get the output.
    Each method is essentially a step in pipeline of Form Processing. """

    def __init__(self,image):
        """Initialize all the attributes required,
        Args: Image Path"""
        if isinstance(image,np.ndarray):
            self.image = image
            self.impath = "input.png"
            cv2.imwrite(self.impath,image)
        else:
            self.image = cv2.imread(image)#read Image
            self.impath = image
            print(self.image.shape)
        self.segdir = "./segments"
        self.roi_list = []
        self.predictions = []#contains all the predictions in the list
        self.checkpoints = "./models/model.pth"# Checkpoints path


    def frcnn(self):
        """The first step in the pipeline is the Input image is passed through Faster RCNN and get the Reigions of Intrests(ROI)
        These regions are extracted and saved in segemnts directory.We use Luminoth python library to predict using the Faster RCNN.
        Args: Self"""

        if not os.path.exists(self.segdir):# Check if dir exists
            os.makedirs(self.segdir)
        else:
            shutil.rmtree(self.segdir)
            os.makedirs(self.segdir)#create segements dir
        temp = self.image.copy()
        command = "lumi predict -c config.yml {}".format(self.impath)# predict command whic is run on terminal
        cmd = command.split(" ")
        output = subprocess.run( cmd, stdout=subprocess.PIPE )# we get the out put of the command from terminal
        js = output.stdout.decode('utf-8').split("\n")#decode terminal output
        bbox_string = js[len(js) - 2]
        print(bbox_string)

        bbox_dict= yaml.load(bbox_string)#use yaml to load a string as a dictionary
        bbox_list = bbox_dict["objects"]
        i=0
        for bbox in bbox_list:
            coords = bbox["bbox"]
            cv2.rectangle(temp,(coords[0],coords[1]),(coords[2],coords[3]),(255,255,255),1)#draw a rectangle with white borders
            roi = temp[coords[1]:coords[3],coords[0]:coords[2]]#use coordinates in boounding box to get Reigons of Intrest.
            filename = "roi{0}.png".format(i)
            self.roi_list.append(filename)
            write_dir = os.path.join(self.segdir,filename)
            cv2.imwrite(write_dir,roi)#save regions of Intrest.
            i=i+1

    def charseg(self):
        """This is the next step in the Pipline we now segment characters from the extracted reigons of intrests using Image processing techiniques
        Args :Self"""

        for file in os.listdir(self.segdir):
            if file.endswith(".png"):
                dirname = os.path.splitext(file)[0]
                rootdir = self.segdir
                write_dir = os.path.join(rootdir,dirname)
                img = cv2.imread(os.path.join(rootdir,file))

                if not os.path.exists(os.path.join(rootdir,dirname)):
                    os.makedirs(os.path.join(rootdir,dirname))
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#convert image to grayscale

                blur = cv2.GaussianBlur(gray,(5,5),0)#blur the image using gaussian binary ,helps in noise removal
                ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#binarize the image

                ctrs, hier = cv2.findContours(th3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#find countours in the Image
                sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])#Sort the countours
                for i, ctr in enumerate(sorted_ctrs):
                    x, y, w, h = cv2.boundingRect(ctr)#get the bounding rectangle

                    if w>20 and h>20 and w<70 and h<70:
                        char = img[y:y+h, x:x+w]#extract character
                        cv2.rectangle(img,(x,y),( x + w, y + h ),(255,255,255),2)
                        cv2.imwrite(os.path.join(write_dir,'{}.png'.format(i)),char)#Save character in each roi directory






    def predict(self):
        """This method is uses the model in model.py file to predict the classes in segmented character Args:Self"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )#check the divice on the computer
        net = Classifier()#instantiate the Classifier
        net.load_state_dict(torch.load(self.checkpoints,map_location = device))#load the model weights

        loader = transforms.Compose([transforms.Resize(28),transforms.ToTensor(),#define the transforms to performed before prediction
                  transforms.Normalize((0.1307,), (0.3081,)),])#we convert to tensor and Normalize
        net.to(device)#transfer model to device
        net.eval()# sset the model to evaluation mode
        dir_list = os.walk(self.segdir)#get subdirectory list
        # print(dir_list)
        for dir in os.walk(self.segdir):
            subdir = dir[0]
            if subdir == self.segdir:
                pass
            else:
                num = []
                sort_file = []
                print(subdir)
                for file in os.listdir(subdir):#find and append all images in subdir to a list
                    if file.endswith(".png"):
                        fstring = os.path.splitext(file)[0]
                        fnum = int(fstring)
                        sort_file.append(fnum)

                sort_file.sort()#sort all the images in the subdirectory by filename
                for i in sort_file:
                    file = "{}.png".format(i)



                    filename = os.path.join(subdir,file)#create filename

                    img = cv2.imread(filename)#read file
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#to grayscale
                    img = np.invert(img)#invert image for to make image similar to dataset
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            if img[i][j] < 8:#if pixel value is less than 8 turn it to zer0,it is noise removal technique
                                img[i][j]=0

                    im = Image.fromarray(img)#read into PIL image
                    im = im.resize((28,28))#resize Image
                    image = loader(im).float()#perform transforms on the Image
                    image = Variable(image, requires_grad=True)#load test Image
                    image = image.unsqueeze(0)
                    image = image.to(device)#send the image to device
                    output = net(image)#predict
                    pred = output.max(1, keepdim=True)[1]
                    num.append(pred[0][0].item())#get class name
                    self.predictions.append(num)
        # self.predictions = list(dict.fromkeys(self.predictions))
        self.predictions = sorted(self.predictions)
        self.predictions = list(self.predictions for self.predictions,_ in itertools.groupby(self.predictions))#remove duplicates of the classname


    def output(self):
        """ This is the final method which takes the predicitons and pritns them out onto the screen and writes them to an outfile"""
        self.frcnn()#call frcnn method
        self.charseg()#call charseg method
        self.predict()#call predict method
        print("\n")
        print("---------------------------------------------------------")
        if os.path.exists("./outputs/out.csv"):
            outfile = open('./outputs/out.csv', 'a+')
        else:
            outfile = open('./outputs/out.csv', 'w')#open csv
        with outfile:
            writer = csv.writer(outfile)
            for i in self.predictions:
                if len(i) <10:
                    date = " "
                    for j in range(len(i)):
                        date += str(i[j])#concate to string

                    out_row = [self.impath,"date",date]#write to the csv file

                    writer.writerows(out_row)
                    print("Date : {}".format(date))
                else:
                    acno = " "
                    for j in range(len(i)):
                        acno += str(i[j])

                    out_row = [self.impath,"A/C no",acno]#write to csv file

                    writer.writerows(out_row)

                    print("A/C no : {}".format(acno))#print to screen
        print("---------------------------------------------------------")
