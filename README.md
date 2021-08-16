
## WHAT YOU CAN FIND: (to be updated)	

* *eye_blinking_detection_p2.py* : testing code
* *shape_predictor_68_face_landmarks.dat* : needed to run the code
* *monitor_eye_care_system_using_blink.pdf* : paper about different techinques for blinking detection

## Last works:
* *get_landmarks*: used to get the face landmarks of a given image or from webcam
* *NN*: folder with the NN implemented so far (naive approach)
	* *NaiveNotebook.ipynb*: notebook with fancy utility functions for training a CNN with the 'dataset_b_Eye_images'
	* *naive_approach.py*: python code, same as the notebook, but without the fancy utilities
		* So far, a CNN training with 'analyzedMrlEyes' is implemented
		* ```python3 naive_approach.py x [--load]```
			* ```x```: either 1 - for 'dataset_b_Eye_images' model - or 2 - for 'analyzedMrlEyes'
			* ```--load``` option is used to load an already created numpy arrays dataset. If not used, a new dataset will be built
		* For performance reasons, 25% of the images (randomly chosend) were loaed and used during the training
	* Note: ```tensorflow-gpu``` has been used. If not present in the running machine, it will use ```tensorflow``` instead.

## Notes:
All programs were developed and tested on Ubuntu OS. Other operating systems may have some problems.
