#!/usr/bin/env python3

import rospy
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def read_data():
	image_paths = []
	poses = []
	PATH = '/home/shania/catkin_ws/src/android/scripts/datasets/TurtlebotTourCVSSP/'
	file_name = 'metadata.txt'
	with open("/".join([PATH, file_name])) as f:
		lines = f.readlines()
	for i in range(3, len(lines)):  # skipping first 3 lines
		data = lines[i].split()
		image_paths.append("/".join([PATH, data[0]]))
		poses.append([float(x) for x in data[1:]])
    
	return image_paths, poses
        
        
def create_path_message():
	# Initialize a new Path message
	path_msg = Path()    	
	
		# Extract poses and add them to the path
	_, poses = read_data()
	for i in range(len(poses)):
		# print(poses[i])
		pose_msg = PoseStamped()
		pose_msg.header.stamp = rospy.Time.now()  # Set the timestamp
		pose_msg.header.frame_id = "map-gt" 
		pose_msg.pose.position.x = poses[i][0]  # X coordinate
		pose_msg.pose.position.y = poses[i][1]  # Y coordinate
		pose_msg.pose.position.z = poses[i][2]  # Z coordinate
		# You can set orientation if needed
		pose_msg.pose.orientation.x = poses[i][3]
		pose_msg.pose.orientation.y = poses[i][4]
		pose_msg.pose.orientation.z = poses[i][5]
		pose_msg.pose.orientation.w = poses[i][6]
        
        # Add the pose to the path
		# print(pose_msg)
		path_msg.poses.append(pose_msg)
	
	path_msg.header.stamp = rospy.Time.now()  # Set the timestamp
	path_msg.header.frame_id = "map-gt" 
	
	# Apply transform to path message	
	# t_msg = listener.transformPose("map", path_msg)  
	# t_msg.header.stamp = rospy.Time.now()
    
	return path_msg


if __name__ == '__main__':
	rospy.init_node('path_publisher')
    
	# Create a publisher for the path message
	path_publisher = rospy.Publisher('/path_topic', Path, queue_size=10)
    
	rate = rospy.Rate(1)  # 1 Hz
    
	while not rospy.is_shutdown():
		# Create and publish the path message
		# listener = tf.TransformListener()  # Don't need this as a transform is running, all we need is to state the original frame when publishing
		path_msg = create_path_message()
		path_publisher.publish(path_msg)
		rospy.loginfo("Path message published.")
		rate.sleep()

