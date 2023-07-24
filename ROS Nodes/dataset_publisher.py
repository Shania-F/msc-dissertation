#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError   # sudo apt-get install cv_bridge
from PIL import Image as Img  # pip install Pillow
import numpy
import random


def read_data():
        image_paths = []
        poses = []
        PATH = '/home/shania/catkin_ws/src/android/scripts/KingsCollege'
        file_name = 'dataset_test.txt'
        with open("/".join([PATH, file_name])) as f:
            lines = f.readlines()
        for i in range(3, len(lines)):  # skipping first 3 lines
            data = lines[i].split()
            image_paths.append("/".join([PATH, data[0]]))
            poses.append([float(x) for x in data[1:]])

        return image_paths, poses        


def photo_publisher():
	bridge = CvBridge()
	pub = rospy.Publisher('/photo', Image, queue_size=10)  # topic name
	rospy.init_node('dataset_publisher') 
	rate = rospy.Rate(0.1) 
	while not rospy.is_shutdown():
		image_paths, poses = read_data()
		n = random.randint(0, len(image_paths))
		img = Img.open(image_paths[n])
		print("True Pose: ", poses[n])

		np_img = numpy.array(img)
		# convert to ros format
		msg = bridge.cv2_to_imgmsg(np_img, "rgb8")
		pub.publish(msg)
		rate.sleep()


if __name__ == '__main__':
	try:        
		photo_publisher()
	except rospy.ROSInterruptException:
		pass
        
