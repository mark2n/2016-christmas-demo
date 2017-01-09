/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015, Daichi Yoshikawa
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Daichi Yoshikawa nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Daichi Yoshikawa
 *
 *********************************************************************/

#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include <ros/ros.h>
#include "ahl_utils/exception.hpp"
#include "ahl_robot/robot/parser.hpp"
#include "ahl_robot/robot/tf_publisher.hpp"
#include "ahl_digital_filter/pseudo_differentiator.hpp"

// #include "ahl_robot_samples/mobility/mecanum_wheel.hpp"

#include <sensor_msgs/JointState.h>
#include <geometry_msgs/Twist.h>
#include <brics_actuator/JointPositions.h>
#include "youbot_torque_measure/Force_EndEffector.h"
#include "youbot_torque_measure/Force_test.h"

#include <boost/units/io.hpp>
#include <boost/units/systems/si/length.hpp>
#include <boost/units/systems/si/plane_angle.hpp>
#include <boost/units/systems/si/velocity.hpp>
#include <boost/units/systems/si/torque.hpp>
#include <boost/units/systems/si/force.hpp>
#include <boost/units/systems/angle/degrees.hpp>
#include <boost/units/conversion.hpp>

// for testing the gravity compensation
#include "ahl_robot/ahl_robot.hpp"
#include "ahl_robot/definition.hpp"
#include "ahl_robot_controller/robot_controller.hpp"
#include "ahl_robot_controller/tasks.hpp"

#include <deque>
#include <gazebo-5.3/gazebo/math/Helpers.hh>

// for mecanum
// #include "ahl_robot_samples/mobility/mecanum_wheel.hpp"

using namespace ahl_robot;
using namespace ahl_ctrl;
using namespace youbot_torque_measure;
// using namespace ahl_rsample;

// DOFs
u_int32_t micro_dof = 5;
u_int32_t base_dof = 4;
u_int32_t macro_dof = 3;

// virtual Manipulator
Eigen::VectorXd q_mnp_virtual(macro_dof + micro_dof);
Eigen::VectorXd dq_mnp_virtual(macro_dof + micro_dof);

// base
Eigen::Vector4d q_base = Eigen::Vector4d::Zero();
Eigen::Vector4d dq_base = Eigen::Vector4d::Zero();
Eigen::Vector4d torque_base = Eigen::Vector4d::Zero();

Eigen::Vector3d q_macro_mnp = Eigen::Vector3d::Zero();
Eigen::Vector3d dq_macro_mnp = Eigen::Vector3d::Zero();
Eigen::Vector3d torque_macro_mnp = Eigen::Vector3d::Zero();

// arm
Eigen::VectorXd q_micro_mnp(micro_dof);
Eigen::VectorXd dq_micro_mnp(micro_dof);
Eigen::VectorXd torque_micro_mnp(micro_dof);

// jacobian, force, torque
double torqueBias = 0.3;
bool COMPLIANT = false;
Eigen::VectorXd force_external(6); // x, y, z, roll, pitch, yaw; therefore 6
Eigen::MatrixXd Jacobian; // jacobian of end effector to base_x link
Eigen::VectorXd torque_compensated(macro_dof + micro_dof);
Eigen::VectorXd torque_gravity_compensation(macro_dof + micro_dof);
Eigen::VectorXd torque_measured(macro_dof + micro_dof);

// conditions
bool ONLY_ARM = true;

// publisher, message
ros::Publisher pub_force_endeffector;
ros::Publisher pub_force_test;
Force_test msg_external_force;

ros::Publisher armPositionCommandPublisher;
brics_actuator::JointPositions command;

ros::Publisher pub_move_base;
geometry_msgs::Twist command_move_base;

// robot model
std::string name = "youbot";
RobotPtr robot = std::make_shared<Robot>(name);

// Manipulator model
const std::string mnp_name = "mnp";
ManipulatorPtr mnp;
RobotControllerPtr controller;
TaskPtr gravity_compensation;

// mecanum wheel
const double tread_width  = 0.3;
const double wheel_base   = 0.471;
const double wheel_radius = 0.05;
double l1 = 0.5 * tread_width;
double l2 = 0.5 * wheel_base;
Eigen::MatrixXd decomposer;
Eigen::MatrixXd decomposer_;
Eigen::MatrixXd composer;
Eigen::MatrixXd composer_;

double bias_x = 5;
double bias_y = 0.0;
double threshold_x = 1.3;
double threshold_y = 0.8;

double SPEED_P = 0.15;
double SPEED_N = 0 - SPEED_P;

int buffer_size = 20;

std::deque<double> buffer_x;
std::deque<double> buffer_y;
bool CONTACT_X = false;
bool CONTACT_Y = false;



Eigen::MatrixXd pInv(Eigen::MatrixXd& in)
{
    // Moore–Penrose pseudo inverse of a matrix
  Eigen::MatrixXd in_T;
  Eigen::MatrixXd out;
  in_T = in.transpose();
  
  Eigen::MatrixXd middle;
  middle =  in * in_T;
  if (middle.determinant()==0) return in_T;
  
  out = in_T * middle.inverse();
  return out; 
}

Eigen::MatrixXd mod_Jacobian(Eigen::MatrixXd& in)
{
  // modify jacobian so that the dominating external force is more obvious
  
  double sum_abs = 0;

  for (int i = 0; i < in.rows(); i ++)
  {
    sum_abs = in.row(i).cwiseAbs().sum(); // sum of each row
    
    for (int j = 0; j < in.cols(); j ++)
    {
      if (fabs(in(i, j)) < double(sum_abs / in.cols()) ) in(i, j) = 0;
      
    } 
    
  }
  
  Eigen::MatrixXf::Index index_max_row, index_max_col;
  double temp = 0;
  for (int j = 0; j < in.cols(); j ++) // fitler columns 
  {
    temp = in.col(j).cwiseAbs().maxCoeff(&index_max_row, &index_max_col);
    temp = in(index_max_row, j);
    in.col(j).setZero();
    in(index_max_row, j) = temp;
  }
  
  return in;
  
}
  
void jointStatesCallback_hardcodded(const sensor_msgs::JointState::ConstPtr& msg)
{// use the joint states of the arm to parse the force at the end-effector
  // hard coded for the jacobian matrix, as the arm is fix. sothat the computational load is less.
  
  Eigen::MatrixXd hard_coded_jacobian;
hard_coded_jacobian.resize(6,5);
hard_coded_jacobian << 
	0,         0,         0,         0,         0,
        0,         0,         0,         0,         0,
        0,         0,         0,         0,         0,
        0,         0,         0,         0, -0.902889,
        0, -0.983001, -0.983001, -0.983001,         0,
        1,         0,         0,         0,         0;
	
  Eigen::VectorXd hard_coded_gravity_compensation(8, 1);
  hard_coded_gravity_compensation << 0, 0, 0, 0, -0.210732, 0.0147815, -0.119112, -0.000477605;
  
  
  if (strncmp(msg->name[0].c_str(), "wheel", 5) == 0 && !ONLY_ARM) 
  { 
    // update the wheel states
    for (int i = 0; i < base_dof; i++)
    {
      q_base(i) = msg->position[i];
      dq_base(i) = msg->velocity[i];
      torque_base(i) = msg->effort[i];      
    }
    
    // convert wheels' states into 3 virtual joints
    q_macro_mnp = composer * q_base;
    dq_macro_mnp = composer * dq_base;
    torque_macro_mnp = composer_.transpose() * torque_base; // TODO: is this correct ???
    std::cout << torque_macro_mnp << std::endl << "********" << std::endl;
  }
  else
  {
   torque_macro_mnp = Eigen::Vector3d::Zero();
  }
  

  if (strncmp(msg->name[0].c_str(), "arm", 3) == 0) 
  {    
    // update arm states
    for (int i = 0; i < micro_dof; i++)
    {
      torque_micro_mnp(i) = msg->effort[i];
    }
  }
  else
  {
    return;
  }

  // compose the whole manipulator states
  torque_measured << torque_macro_mnp, torque_micro_mnp;
  
  torque_compensated = torque_measured + hard_coded_gravity_compensation;
  
  force_external = hard_coded_jacobian * torque_compensated.bottomRows(5);   
  
  /*
  Eigen::MatrixXd T_transform;
  T_transform = mnp->getTransformAbs(8);*/
  
  msg_external_force.header.stamp = ros::Time::now();
 
  if ( buffer_x.size() < buffer_size )
  {
    buffer_x.push_back(force_external(4)); // firstly fill the buffer
    return;
  }
  else
  {
    buffer_x.push_back(force_external(4)); // when the buffer is full, kepp it refreshed
    buffer_x.pop_front();
  };
  
  if ( buffer_y.size() < buffer_size )
  {
    buffer_y.push_back(force_external(5)); // firstly fill the buffer
    return;
  }
  else
  {
    buffer_y.push_back(force_external(5)); // when the buffer is full, kepp it refreshed
    buffer_y.pop_front();
  };
  
  if ( buffer_y.size() < buffer_size )
  {
    buffer_y.push_back(force_external(5)); // firstly fill the buffer
    return;
  }
  else
  {
    buffer_y.push_back(force_external(5)); // when the buffer is full, kepp it refreshed
    buffer_y.pop_front();
  };
  
  double mean_buffer = 0;
  
  // pitch force -> x direction movement
  for (int i = 0; i < buffer_size; ++i) mean_buffer += buffer_x[i];
  mean_buffer = double( mean_buffer / buffer_size );
  
  if (CONTACT_X && (fabs(mean_buffer - bias_x) < threshold_x)) CONTACT_X = false;
  if (!CONTACT_X && (fabs(mean_buffer - bias_x) > threshold_x)) CONTACT_X = true;
  double speed_x = 0;
  CONTACT_X? (mean_buffer > bias_x ? (speed_x = SPEED_P) : (speed_x = SPEED_N) ) : (speed_x = 0);
  
//   std::cout <<"contact X ? " << CONTACT_X << " & " << speed_x << std::endl;
  
  
  // yaw force -> y direction movement
  mean_buffer = 0;
  for (int i = 0; i < buffer_size; ++i) mean_buffer += buffer_y[i];
  mean_buffer = double( mean_buffer / buffer_size );
  
  if (CONTACT_Y && (fabs(mean_buffer - bias_y) < threshold_y)) CONTACT_Y = false;
  if (!CONTACT_Y && (fabs(mean_buffer - bias_y) > threshold_y)) CONTACT_Y = true;
  double speed_y = 0;
  CONTACT_Y? (mean_buffer > bias_y ? (speed_y = SPEED_P * 3) : (speed_y = SPEED_N * 3) ) : (speed_y = 0);
  
  command_move_base.linear.x = speed_x;
//   command_move_base.linear.y = speed_y;
  command_move_base.angular.z = speed_y;
  pub_move_base.publish(command_move_base);
//   std::cout <<"contact Y ? " << CONTACT_Y << " & " << mean_buffer - bias_y  << std::endl;
  
}

void jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg)
{// use the joint states of the arm to parse the force at the end-effector
  
  if (strncmp(msg->name[0].c_str(), "wheel", 5) == 0 && !ONLY_ARM) 
  { 
    // update the wheel states
    for (int i = 0; i < base_dof; i++)
    {
      q_base(i) = msg->position[i];
      dq_base(i) = msg->velocity[i];
      torque_base(i) = msg->effort[i];      
    }
    
    // convert wheels' states into 3 virtual joints
    q_macro_mnp = composer * q_base;
    dq_macro_mnp = composer * dq_base;
    torque_macro_mnp = composer_.transpose() * torque_base; // TODO: is this correct ???
//     std::cout << torque_macro_mnp << std::endl << "********" << std::endl;
  }
  else
  {
   q_macro_mnp = Eigen::Vector3d::Zero();
   dq_macro_mnp = Eigen::Vector3d::Zero();
   torque_macro_mnp = Eigen::Vector3d::Zero();
  }
  

  if (strncmp(msg->name[0].c_str(), "arm", 3) == 0) 
  {    
    // update arm states
    for (int i = 0; i < micro_dof; i++)
    {
      q_micro_mnp(i) = msg->position[i];
      dq_micro_mnp(i) = msg->velocity[i];
      torque_micro_mnp(i) = msg->effort[i];
    }
  }
  else
  {
    return;
  }

  // compose the whole manipulator states
  q_mnp_virtual << q_macro_mnp, q_micro_mnp;
  dq_mnp_virtual << dq_macro_mnp, dq_micro_mnp;
  torque_measured << torque_macro_mnp, torque_micro_mnp;
  
  // update robot
  robot->update(q_mnp_virtual);
  robot->computeJacobian(mnp_name);
  robot->computeMassMatrix();

  Jacobian = mnp->getJacobian()[mnp->getDOF()];
  // Jv = robot->getJacobian(mnp_name); // by default, the jacobian is from base (base_virtual) to the end effector
  
  //       forceAtEndeffector = Jv.rightCols(arm_dof).transpose().colPivHouseholderQr().solve(torque);
  //       F_eef = Jv.transpose().bottomRows(5).colPivHouseholderQr().solve(torque);
  controller->updateModel();
  controller->computeGeneralizedForce(torque_gravity_compensation);
  
  torque_compensated = torque_measured + torque_gravity_compensation;
  //       F_eef = Jv.transpose().bottomRows(5).fullPivLu().solve(torque);
  //       F_eef = Jv.transpose().bottomRows(5).fullPivLu().solve(torque_compensated); 
  //   std::cout << Jacobian.rightCols(5) << "********" << std::endl;
//   std::cout << Jacobian << "********" << std::endl;
  //   F_eef = Jv.rightCols(5) * torque_compensated;   
//   force_external = Jacobian * torque_compensated;   
  Eigen::MatrixXd jacobian_short = Jacobian.rightCols(5);
  Eigen::MatrixXd jacobian_mod = mod_Jacobian(jacobian_short);
  std::cout << "*******" << std::endl << torque_gravity_compensation << std::endl ;
  force_external = jacobian_mod * torque_compensated.bottomRows(5);   
  
  /*
  Eigen::MatrixXd T_transform;
  T_transform = mnp->getTransformAbs(8);*/
  
  msg_external_force.header.stamp = ros::Time::now();
  
//   msg_external_force.force_1 = force_external(0);
//   msg_external_force.force_2 = force_external(1);
//   msg_external_force.force_3 = force_external(2);
//   msg_external_force.force_4 = force_external(3);
//   msg_external_force.force_5 = force_external(4);
//   msg_external_force.force_6 = force_external(5);
//   
//   pub_force_test.publish(msg_external_force);
  // deal with contact detection and force interpolation  
  if ( buffer_x.size() < buffer_size )
  {
    buffer_x.push_back(force_external(4)); // firstly fill the buffer
    return;
  }
  else
  {
    buffer_x.push_back(force_external(4)); // when the buffer is full, kepp it refreshed
    buffer_x.pop_front();
  };
  
  if ( buffer_y.size() < buffer_size )
  {
    buffer_y.push_back(force_external(5)); // firstly fill the buffer
    return;
  }
  else
  {
    buffer_y.push_back(force_external(5)); // when the buffer is full, kepp it refreshed
    buffer_y.pop_front();
  };
  
  if ( buffer_y.size() < buffer_size )
  {
    buffer_y.push_back(force_external(5)); // firstly fill the buffer
    return;
  }
  else
  {
    buffer_y.push_back(force_external(5)); // when the buffer is full, kepp it refreshed
    buffer_y.pop_front();
  };
  
  double mean_buffer = 0;
  
  // pitch force -> x direction movement
  for (int i = 0; i < buffer_size; ++i) mean_buffer += buffer_x[i];
  mean_buffer = double( mean_buffer / buffer_size );
  
  if (CONTACT_X && (fabs(mean_buffer - bias_x) < threshold_x)) CONTACT_X = false;
  if (!CONTACT_X && (fabs(mean_buffer - bias_x) > threshold_x)) CONTACT_X = true;
  double speed_x = 0;
  CONTACT_X? (mean_buffer > bias_x ? (speed_x = SPEED_P) : (speed_x = SPEED_N) ) : (speed_x = 0);
  
//   std::cout <<"contact X ? " << CONTACT_X << " & " << speed_x << std::endl;
  
  
  // yaw force -> y direction movement
  mean_buffer = 0;
  for (int i = 0; i < buffer_size; ++i) mean_buffer += buffer_y[i];
  mean_buffer = double( mean_buffer / buffer_size );
  
  if (CONTACT_Y && (fabs(mean_buffer - bias_y) < threshold_y)) CONTACT_Y = false;
  if (!CONTACT_Y && (fabs(mean_buffer - bias_y) > threshold_y)) CONTACT_Y = true;
  double speed_y = 0;
  CONTACT_Y? (mean_buffer > bias_y ? (speed_y = SPEED_P * 3) : (speed_y = SPEED_N * 3) ) : (speed_y = 0);
  
  command_move_base.linear.x = speed_x;
//   command_move_base.linear.y = speed_y;
  command_move_base.angular.z = speed_y;
  pub_move_base.publish(command_move_base);
//   std::cout <<"contact Y ? " << CONTACT_Y << " & " << mean_buffer - bias_y  << std::endl;
  
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "parser_test");
  ros::NodeHandle nh;
  
  ParserPtr parser = std::make_shared<Parser>();
  std::string path = "/home/tianbao/Projects/catkin_ws/src/ahl_wbc/wbc/ahl_robot/yaml/youbot.yaml";
  parser->load(path, robot);
  
  controller = std::make_shared<RobotController>();
  controller->init(robot);
  
  mnp = robot->getManipulator("mnp");
  
  gravity_compensation = std::make_shared<GravityCompensation>(robot);
  controller->addTask(gravity_compensation, 0);
  
  // initialise the arm position
  armPositionCommandPublisher = nh.advertise<brics_actuator::JointPositions > ("/arm_1/arm_controller/position_command", 1);
  std::vector <brics_actuator::JointValue> armJointPositions;
  armJointPositions.resize(micro_dof);
  std::vector <double> joint_value_vector(micro_dof);
  joint_value_vector = {2.9569536007408317, 0.4739575839886909, -0.9624897412803051, 2.465840498581014, 2.8967032938532764};
  std::stringstream joint_name;
  
  command.positions = armJointPositions;
  
  for (int i = 0; i < micro_dof; ++i )
  {
    joint_name.str("");
    joint_name << "arm_joint_" << i + 1;
//     armJointPositions[i].timeStamp = ros::Time::now();
    command.positions[i].value = joint_value_vector[i];
    command.positions[i].joint_uri = joint_name.str();
    command.positions[i].unit = boost::units::to_string(boost::units::si::radians);
  }
  
  armPositionCommandPublisher.publish(command);
  ros::Duration(2).sleep();
  armPositionCommandPublisher.publish(command);
  
  std::cout << "ready to move." << std::endl;
  
  // test move base
  pub_move_base = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);
  command_move_base.linear.x = 0;
  command_move_base.linear.y = 0;
  command_move_base.linear.z = 0;
  command_move_base.angular.x = 0;
  command_move_base.angular.y = 0;
  command_move_base.angular.z = 0;
  pub_move_base.publish(command_move_base);
  ros::Duration(1.5).sleep();
  
//   ros::Subscriber sub = nh.subscribe("/joint_states", 1000, jointStatesCallback);
  ros::Subscriber sub = nh.subscribe("/joint_states", 1000, jointStatesCallback);
  pub_force_test = nh.advertise<Force_test>("youbot_arm_force_test", 1000);
  
  ros::spin();

  return 0;
}

// arm posture 
// name: ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4', 'arm_joint_5', 'gripper_finger_joint_l', 'gripper_finger_joint_r']
// position: [2.9569536007408317, 0.4739575839886909, -0.9624897412803051, 2.465840498581014, 2.8967032938532764, 0.0, 0.0]





// #include "ros/ros.h"
// #include "std_msgs/String.h"
// #include <sensor_msgs/JointState.h>
// #include "Force_EndEffector.h"
// 
// #include "Eigen/Dense"
// // #include "robot.hpp"
// 
// ros::Publisher pub_joints_base;
// ros::Publisher pub_joints_arm;
// 
// ros::Publisher pub_force_endeffector;
// 
// const uint16_t dof = 5;
// const uint16_t CScorr = 6;
// Eigen::Matrix<float, CScorr, dof> InvJacobian;
// Eigen::Matrix<float, CScorr, dof> Jacobian;
// 
// Eigen::VectorXd q(dof, 1);
// Eigen::VectorXd q_d(dof, 1);
// 
// Eigen::VectorXd torque(dof, 1);
// Eigen::VectorXd force(3, 1);
// 
// 
// 
// void splitMessages(const sensor_msgs::JointState::ConstPtr& msg)
// {
//   // split and publish messages 
//    if (strncmp(msg->name[0].c_str(), "arm", 3) == 0) 
//   {
//     if (msg->name.size())
//     {
//       for (int i = 0; i < dof; i++)
//       {
// 	 q(i) = msg->position[i];
// 	 q_d(i) = msg->velocity[i];
// 	 torque(i) = msg->effort[i];
//       }
//       
//       ROS_INFO("torque: %f, %f,%f, %f, %f",  torque(0), torque(1), torque(2), torque(3), torque(4));
//     }
//     pub_joints_arm.publish(msg);
//   } 
//   else
//   {
//     if (strncmp(msg->name[0].c_str(), "wheel", 4) == 0)
//     {
//       pub_joints_base.publish(msg);
//     }
//   }
// }
// 
// // void calculateJacobian(RobotPtr& robot)
// // {
// //   // calculate the jacobian of the robot
// // 
// // }
// // 
// // void inverseJacobian()
// // {
// //   // inverse the Jacobian to translate torques at joints to force at the end effoector
// //   if (calculateSDV(Jacobian))
// //   {
// //     // direct inverse
// // 
// //   }else
// //   {
// //     // pseudo inverse
// // 
// //   }
// //   InvJacobian = robot->Jacobian * robot->Jacobian;
// // }
// 
// void jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg)
// {
//   // publish joint states for arm and base separately
//   splitMessages(msg);
//   
//   
// //   // calculate jacobian
// //   calculateJacobian();
// // 
// //   // calculate inverse jacobian
// //   inverseJacobian();
// 
// }
// 
// int main(int argc, char **argv)
// {
//   ros::init(argc, argv, "youbot_torq_measure_node");
// 
//   ros::NodeHandle n;
// 
//   pub_joints_base = n.advertise<sensor_msgs::JointState>("youbot_base_states", 1);
//   pub_joints_arm = n.advertise<sensor_msgs::JointState>("youbot_arm_states", 1);
//   pub_force_endeffector = n.advertise<youbot_torque_measure::Force_EndEffector>("youbot_arm_force_endeffector", 1000);
// 
//   ros::Subscriber sub = n.subscribe("/joint_states", 1000, jointStatesCallback);
// 
//   // TODO: initialise the robot
// //   std::string robot_name = "youbot";
// //   RobotPtr robot = std::make_shared<Robot>(robot_name);
// 
//   // TODO:get the necessary parameters for calculation
// //   ParserPtr parser = std::make_shared<Parser>();
// //   std::string path = "/home/daichi/Work/catkin_ws/src/ahl_ros_pkg/ahl_robot/ahl_robot/yaml/youbot.yaml";
// //   parser->load(path, robot);
// // 
// //   const std::string mnp_name = "arm1";
//   
//   ros::spin();
// 
//   return 0;
// }
