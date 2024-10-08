# Emotion-detection-Model

:robot: The project aims to find the activeness of the students based on their emotions expressed by their faces. Students need to deal with a lot of tasks every day. Most of the students in their college life undergo stress and depression. The face expressions are natural and direct means for human beings to communicate their emotions and intentions. This is where Deep learning provides a solution to find their emotions through their faces which can be used to provide them with proper guidance and assistance.

Emotions Detection :neutral_face: :smile: :worried: :fearful: :open_mouth: :angry: :roll_eyes:
--
The emotions shown by common human can be classified into 7 i.e. Happy, Sad, 
Disgust, Fear, Neutral, Angry and Surprise. </br> These emotions Happy, Neutral and 
Surprise are classified as active and the remaining as inactive.  

The facial expressions are analyzed and classified using 2 methodologies: </br>
1. &nbsp; Using Deep Face framework </br>
2. &nbsp; Building a 4-layer sequential CNN model </br>
The Deep Face framework classified the images more accurately when compared 
to the sequential model. </br>

--> Deep Face framework- Deepface is a lightweight face recognition and facial 
attribute analysis framework for python. It is a hybrid framework wrapping various 
state of models like VGG-Face, OpenFace, Google FaceNet, Facebook DeepFace, 
Dlib, ArcFace and SFace.</br> </br>
--> Sequential CNN model- Sequential is the easiest way to build a model in Keras. It 
allows you to build a model layer by layer. The ‘add()’ function is used to add layers 
to the model.</br>

<li>Programming language used: Python 3.9.6</li>
<li>Programming platforms used: Visual Studio Code, Kaggle Kernel</li>
<li>Other software: Excel for storing status of activeness</li></br>

In both the activeness detection systems built using 2 models, the common 
steps carried out were Face detection, recognition of the faces and then 
classifying the emotions shown on the faces. The DeepFace framework 
works better with 97% and here the faces were recognized using Face 
recognition library and then the emotion was classified. The CNN model was 
developed with the famous FER 2013 dataset which contains images 
showing 7 different emotions such as happy, sad, angry, neutral, surprise, 
fear and disgust. The 4-layer CNN model was developed and stored as a h5 
file. Then the h5 model file was used to classify the faces using webcam. 
Here the haarcascade xml file is used to detect the faces and then the 
developed CNN model is used classify the status of activeness. This model 
attained an accuracy of 60%. Thus, the DeepFace model is the better among 
the two models. 
