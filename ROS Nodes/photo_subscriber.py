#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def callback(data):
	print("Received data of type: ", type(data))
	br = CvBridge()
	data = br.imgmsg_to_cv2(data, "bgr8")
	print("Converted to type: ", type(data))
	cv2.imshow("Smartphone Image", data)
	
	cv2.waitKey(0)
	 

if __name__ == '__main__':
	rospy.init_node("photo_subscriber")
	sub = rospy.Subscriber("/photo", Image, callback)
	rospy.loginfo("Photo Node started")
	
	while not rospy.is_shutdown():
		rospy.spin()
	cv2.destroyAllWindows()
	
