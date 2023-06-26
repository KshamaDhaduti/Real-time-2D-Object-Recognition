This project focuses on real-time 2D object recognition using computer vision techniques. The objective is to develop a system that can identify a specified set of objects placed on a white surface from a camera looking straight down, irrespective of translation, scale, or rotation. The system should be able to recognize single objects placed in the image and output an image indicating the identified object. Real-time performance is desired, but if not feasible, a directory of images can be processed instead. The implementation involves thresholding, morphological filtering, connected components analysis, feature computation, training data collection, classification, and evaluation.

----Video:

https://northeastern-my.sharepoint.com/:v:/r/personal/dhaduti_k_northeastern_edu/Documents/obj_recognition.mp4?csf=1&web=1&e=hJ7j9J

-was facing technical issues capturing a lengthy video.

----Wiki report:

https://wiki.khoury.northeastern.edu/pages/resumedraft.action?draftId=136594918&draftShareId=6aa2a584-0625-4a29-adbc-df3e7a071e96&

----Operating System: Windows
    IDE: Visual Studio

----Instructions for Execution

	1. Training Model
		1. Press q to capture a frame 
		2. Press q to capture the feature vectors of that frame
		3. Press q to Quit
	2. Classifier Mode
		1. Press q to capture a frame
		2. Press q for feature matching of obj with data base
		3. Press q to Quit
