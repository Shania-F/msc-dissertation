#!/usr/bin/env python3

import socket
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError   # sudo apt-get install cv_bridge
from PIL import Image as Img  # pip install Pillow
import numpy
import os

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('', 12345))

bridge = CvBridge()


def photo_publisher():
    pub = rospy.Publisher('/photo', Image, queue_size=10)  # topic name
    rospy.init_node('photo_publisher')  # node name
    rate = rospy.Rate(10)  # Hz: loop is run 10 times per second
    while not rospy.is_shutdown():
        server.listen(1)
        print("Listening...")
        
        # if rospy.is_shutdown():  # need to check again

        client_socket, client_address = server.accept()
        print("Connected with {}!".format(client_address))

        file = open('rsa_image_12345.jpg', "wb")

        image_chunk = client_socket.recv(8192)
        while image_chunk:
            # print("...writing")
            file.write(image_chunk)
            image_chunk = client_socket.recv(8192)

        file.close()
        client_socket.close()

        # convert to cv mat for cv_bridge
        img = Img.open("rsa_image_12345.jpg")
        if img:
            print("Image received.")
        np_img = numpy.rot90(numpy.array(img), 3)  # temp fix of rotation issue
        os.remove("rsa_image_12345.jpg")

        # convert to ros format
        msg = bridge.cv2_to_imgmsg(np_img, "rgb8")
        pub.publish(msg)
        rate.sleep()
        


if __name__ == '__main__':
    try:
        photo_publisher()
    except rospy.ROSInterruptException:
        pass
        
