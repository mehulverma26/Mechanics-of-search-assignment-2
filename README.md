## Installing the modules

### you would need Python version >=3.10 to run this
<br>
If you have multiple python versions installed then use this command:

```bash
py -<python version> -m pip install -r requirements.txt
```

else you can use the following command:

```bash
pip install -r requirements.txt
```

## Downloaded this pretrained model for object detection using openCV
<br>

[MobileNet-SSD_deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt)
<br>

[MobileNet-SSD_deploy.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel)

## Prerequisites

1. Google API key (you can get this from the credential page)
2. Google CX (can be taken if you make a search engine on google)
3. Unsplash API key (have to create an app on the website for this as well)

Keep these keys in a ' .env ' file. The .env should look somewhat like this:

```.env
GOOGLE_API_KEY=<replace with your api key>
GOOGLE_CX=<replace with your api key>
UNSPLASH_ACCESS_KEY=<replace with your api key>
```

## Running the program

Run the [<b>v4(main).py</b>](https://github.com/mehulverma26/Mechanics-of-search-assignment-2/blob/main/v4%20(main).py) because that is most up to date code but I am including earlier versions of the code just in case anyone wants to study or take a reference from them otherwise it will be reminder for my journey leading up to the finalized version 😊