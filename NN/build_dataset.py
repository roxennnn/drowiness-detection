import numpy as np
import os
import shutil
from random import randrange, seed
import time

now = int(round(time.time() * 1000))
seed(now)

# dir_path = "../datasets/mrlEyes_2018_01/"
# open_path = "../datasets/mrlEyes/Open/"
# close_path = "../datasets/mrlEyes/Close/"
# leftovers_path = "../datasets/mrlEyes/Leftovers"

# shutil.rmtree("../datasets/mrlEyes")
# os.mkdir("../datasets/mrlEyes")
# os.mkdir(open_path)
# os.mkdir(close_path)
# os.mkdir(leftovers_path)

# for dir in os.listdir(dir_path):
# 	for file in os.listdir(dir_path+dir):
# 		print(file)
# 		filesplit = file.split("_")
# 		# Interesting annotations:
# 		# filesplit[4]: open = 1, close = 0
# 		# filesplit[-2]: bad = 0, good = 1

# 		if filesplit[-2] == "1":
# 			if filesplit[4] == "1":
# 				rnd = randrange(100)
# 				if rnd > 60:
# 					shutil.move(dir_path+dir+"/"+file, open_path+file)
# 				else:
# 					shutil.move(dir_path+dir+"/"+file, leftovers_path)
# 			elif filesplit[4] == "0":
# 				shutil.move(dir_path+dir+"/"+file, close_path+file)
# 			else:
# 				shutil.move(dir_path+dir+"/"+file, leftovers_path)
# 		else:
# 			shutil.move(dir_path+dir+"/"+file, leftovers_path)



open_path = "../datasets/mrlEyesBad/Open/"
close_path = "../datasets/mrlEyesBad/Close/"
dir_path = "../datasets/mrlEyes/Leftovers/"


for file in os.listdir(dir_path):
	print(file)
	filesplit = file.split("_")
	# Interesting annotations:
	# filesplit[4]: open = 1, close = 0
	# filesplit[-2]: bad = 0, good = 1

	if filesplit[-2] == "0":
		if filesplit[4] == "1":
			rnd = randrange(100)
			if rnd > 60:
				shutil.move(dir_path+"/"+file, open_path+file)
		elif filesplit[4] == "0":
			shutil.move(dir_path+"/"+file, close_path+file)