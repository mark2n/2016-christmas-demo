#ifndef __IMS_ROBOT_HPP
#define __IMS_ROBOT_HPP


#include "Eigen/Dense"
namespace youbot
{
  class Robot
  {
  public:
    std::string name_;

  private:
    uint16_t dof_;
    Eigen::MatrixXd q(5,1);
    Eigen::MatrixXd q_d(5,1);
      
    Eigen::MatrixXd jacobian(6,5);
    Eigen::MatrixXd invJacobian(6,5);
    
    Eigen::MatrixXd torque(5,1);
    Eigen::MatrixXd force(6,1);
  using RobotPtr = std::shared_ptr<Robot>;
  };  
  
}
#endif