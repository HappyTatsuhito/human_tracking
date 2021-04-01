#!/usr/bin/env python
# -*- coding: utf-8 -*

import rospy
import time
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
# -- ros msgs --
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
# -- custom srvs ---
from mimi_manipulation_pkg.srv import RecognizeCount, RecognizeLocalize, DetectDepth

class MimiControl(object):
    def __init__(self):
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=1)

        self.twist_value = Twist()

    def angleRotation(self, degree):
        while degree > 180:
            degree = degree - 360
        while degree < -180:
            degree = degree + 360
        angular_speed = 50.0 #[deg/s]
        target_time = abs(1.76899*(degree /angular_speed))  #[s]
        if degree >= 0:
            self.twist_value.angular.z = (angular_speed * 3.14159263 / 180.0) #rad
        elif degree < 0:
            self.twist_value.angular.z = -(angular_speed * 3.14159263 / 180.0) #rad
        rate = rospy.Rate(500)
        start_time = time.time()
        end_time = time.time()
        while end_time - start_time <= target_time:
            self.cmd_vel_pub.publish(self.twist_value)
            end_time = time.time()
            rate.sleep()
        self.twist_value.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist_value)


class HumanTracking(object):
    def __init__(self):
        self.mimi_control = MimiControl()
        
        rospy.Subscriber('/camera/color/image_raw', Image, self.realsenseCB)
        rospy.Subscriber('/servo/angle_list', Float64MultiArray, self.motorAngleCB)
        self.ros_image = Image()
        self.head_pub = rospy.Publisher('/servo/head',Float64,queue_size=1)
        self.head_angle = Float64()

        self.count_object = rospy.ServiceProxy('/recognize/count', RecognizeCount)
        self.localize_object = rospy.ServiceProxy('/recognize/localize', RecognizeLocalize)
        self.detect_depth = rospy.ServiceProxy('/detect/depth',DetectDepth)

        cascade_path = '../weight/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def realsenseCB(self, image):
        self.ros_image = image

    def motorAngleCB(self, angle_list):
        self.head_angle = angle_list[5]
        
    def faceToFace(self):
        color_image = bridge.imgmsg_to_cv2(self.ros_image, desired_encoding='bgr8')
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image)
        if not bool(len(faces)):
            self.head_pub.publish(-20.0)
            return
        
        x, y, w, h = faces[0]
        center_x = (y+(y+h))/2
        center_y = (x+(x+w))/2
        detect_res = self.detect_depth(center_x, center_y)
        face_point = detect_res.centroid_point
        face_point.z -= 0.98
        face_angle = math.atan2(face_point.z, face_point.x)/math.pi*180
        if face_angle > 10:
            face_angle += self.head_angle
            if face_angle > 30: face_angle = 30
            if face_angle < -30: face_angle = -30
            self.head_pub.publish(face_angle)
        return
        

    def main(self):
        while not rospy.is_shutdown():
            # 1回目の人検出
            rospy.wait_for_service('/recognize/count')
            count_res = self.count_object('person')
            if not bool(count_res.object_num):
                self.head_pub.publish(0.0)
                rospy.sleep(0.5)
                continue

            # 2回目の人検出
            start_time = time.time()
            rospy.sleep(3.0)
            count_res = self.count_object('person')
            if not bool(count_res.object_num):
                rospy.sleep(0.5)
                continue
        
            # 人の方を向く
            rospy.wait_for_service('/recognize/localize')
            localize_res = self.localize_object('person')
            object_centroid = localize_res.centroid_point
            target_angle = math.atan2(object_centroid.y, object_centroid.x)/math.pi*180
            if target_angle > 10:
                self.mimi_control.angleRotation(target_angle)
            rospy.sleep(4.0)
        
            faceToFace()
        

if __name__ == '__main__':
    rospy.init_node('human_tracking')
    human_track = HumanTracking()
    human_track.main()
    rospy.spin()
