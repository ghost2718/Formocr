import cv2
import numpy as np
import argparse

from formocr import FormOcr#Importing the FormOCr class

#Get Argparse object to take commandline input for the path to Image.
parser = argparse.ArgumentParser()
parser.add_argument("--impath",help = "Give the path to the Image")
args = parser.parse_args()

#Instaniating the FormOcr Object,
fobj = FormOcr(args.impath)
fobj.output()#Use output method to predit the data and A/C no on the form.
