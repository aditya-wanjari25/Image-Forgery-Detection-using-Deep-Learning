from django.shortcuts import render,HttpResponse
# from django.core.files.storage import 
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image
from keras.models import load_model
import numpy as np
from PIL import Image, ImageChops, ImageEnhance



#  Load Model
loaded_model = load_model('model_casia_run1.h5')

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')  #image compression
    image.save(temp_filename, 'JPEG', quality = quality) # fixing the quality parameter to further compress the file size
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image) #Calculating the absolute numerical difference between two images

    extrema = ela_image.getextrema() #Getting the extrema (minimum and maximum pixel values) of the ELA image
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image



image_size = (128, 128)
def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0



class_names = ['Tampered', 'Not Tampered']

# Function to prepare the image for prediction
def prepare_image_for_prediction(image_path):
    # Load and prepare the image
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    return image

# Function to make predictions on a single image
def predict_single_image(image_path, model):
    # Prepare the image for prediction
    image = prepare_image_for_prediction(image_path)
    # Make predictions
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis=1)[0]
    confidence = np.amax(y_pred) * 100
    return class_names[y_pred_class], confidence

# Function to display the image with prediction
def display_image_with_prediction(image_path, model):
    # Load the image
    image = Image.open(image_path)
    # Display the image
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    # Predict the class and confidence
    image_class, confidence = predict_single_image(image_path, model)
    print(f'Prediction: Class: {image_class}, Confidence: {confidence:.2f}%')






# Create your views here.
def index(request):
    if (len(request.FILES) != 0):
      
        fileObj=request.FILES['filePath']
        print(fileObj)
        fileObj=request.FILES['filePath']
        # print(fileObj)
        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        print(filePathName)
        testimage='static/images/'+filePathName
        # imageDisplay = 'static/'+filePathName[1:]
        print(testimage)
        img = Image.open(testimage)
        print(len(img))

    return HttpResponse("This is home page")

def sayHi(request):
    return HttpResponse("Hi Jensy")



def index1(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        print(filename)
        testimage='static/images/'+filename

        #  Read Image from user
        img = Image.open(testimage)
        image_class, confidence = predict_single_image(testimage,loaded_model)
        print(image_class)
        return render(request, 'result.html', {
            'uploaded_file_url': uploaded_file_url,
            'class':image_class
        })
    return render(request, 'home.html')