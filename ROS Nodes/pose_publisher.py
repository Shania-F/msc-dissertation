#!/usr/bin/env python3

# does both: subscribes then publishes

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from PIL import Image as Img
import cv2

import torch
from torchvision import transforms
import torch.nn.functional as F

import models


def image_callback(data):
	# ROS Format to cv2 
	print("Received data of type: ", type(data))
	br = CvBridge()
	data = br.imgmsg_to_cv2(data, "rgb8")
	print("Converted to type: ", type(data), " of shape: ", data.shape)
	# 4032, 1960, 3
	
	# Convert to PIL for Pytorch
	photo = Img.fromarray(data)
	# photo.show()
	
	# Process the image
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.404], [0.229, 0.224, 0.225])
	])
	
	photo = transform(photo)
	print("INPUT: ", photo.shape)
	
	# MODEL
	model = models.ResNet()
	PATH = '/home/shania/catkin_ws/src/android/scripts/model.pth'
	model.load_state_dict(torch.load(PATH))
	model.eval()
		
	pos_out, ori_out = model(photo.unsqueeze(0))
	pos_out = pos_out.squeeze(0).detach().cpu().numpy()
	ori_out = F.normalize(ori_out, p=2, dim=1)
	ori_out = ori_out.squeeze(0).detach().cpu().numpy()
	print(pos_out, ori_out)	
	
	msg = PoseStamped()  # geometry pose stamped
	msg.header.stamp = rospy.Time.now()
	msg.header.frame_id = "pose"
	
	msg.pose.position.x = pos_out[0]
	msg.pose.position.y = pos_out[1]
	msg.pose.position.z = pos_out[2]
	msg.pose.orientation.x = ori_out[0]
	msg.pose.orientation.y = ori_out[1]
	msg.pose.orientation.z = ori_out[2]
	msg.pose.orientation.w = ori_out[3]	
	pub.publish(msg)  
	

if __name__ == '__main__':
	rospy.init_node("pose_publisher")
	
	pub = rospy.Publisher("/camera_pose", PoseStamped, queue_size=10)
	sub = rospy.Subscriber("/photo", Image, callback=image_callback)
	rospy.loginfo("Node has been started")
	
	rospy.spin()
	
