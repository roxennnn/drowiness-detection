# parser to create the annotation files for the eyes extraction NN using the dataset_B_FacialImages Dataset

def parse_and_annotate(filename, destination):
	f = open(filename, "r")
	contents = f.read()
	f.close()
	contents_split = contents.splitlines()
	# print(contents)
	for line in contents_split:
		# print(line)
		line_contents = line.split()
		# print(line_contents)
		new_file = destination + line_contents[0] + ".xml"
		fw = open(new_file, "w+")
		# create the annotation:
		fw.write("<annotation>\n\
	<filename>" + line_contents[0] + "</filename>\n\
	<size>\n\
		<width>100</width>\n\
		<height>100</height>\n\
		<depth>3</depth>\n\
	</size>\n\
	<object>\n\
		<name>eyes</name>\n\
		<bndbox>\n\
			<xmin>" + line_contents[1] + "</xmin>\n\
			<ymin>" + line_contents[2] + "</ymin>\n\
			<xmax>" + line_contents[3] + "</xmax>\n\
			<ymax>" + line_contents[4] + "</ymax>\n\
		</bndbox>\n\
	</object>	\n\
</annotation>")
		fw.close()


parse_and_annotate("dataset_B_FacialImages/EyeCoordinatesInfo_ClosedFace.txt",
					"dataset_B_FacialImages/ClosedFaceAnnotations/")
print("DONE - 1")

parse_and_annotate("dataset_B_FacialImages/EyeCoordinatesInfo_OpenFace.txt",
					"dataset_B_FacialImages/OpenFaceAnnotations/")
print("DONE - 2")
# fw.write("<annotation>\
# 					<folder>Eyes</folder>\
# 					<filename>" + line_contents[0] + "</filename>\
# 					<path>...</path>\
# 					<source>\
# 						<database>Unknown</database>\
# 					</source>\
# 					<size>\
# 						<width>100</width>\
# 						<height>100</height>\
# 						<depth>3</depth>\
# 					</size>\
# 					<segmented>0</segmented>\
# 					<object>\
# 						<name>eyes</name>\
# 						<pose>Unspecified</pose>\
# 						<truncated>0</truncated>\
# 						<difficult>0</difficult>\
# 						<bndbox>\
# 							<xmin>233</xmin>\
# 							<ymin>89</ymin>\
# 							<xmax>386</xmax>\
# 							<ymax>262</ymax>\
# 						</bndbox>\
# 					</object>\
# 				</annotation>\
# 			")