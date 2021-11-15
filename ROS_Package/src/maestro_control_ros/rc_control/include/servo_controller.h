#ifndef SERVO_CONTROLLER_H
#define SERVO_CONTROLLER_H


#include <maestro/Device.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <string.h>

// dynamic reconfig
#include <dynamic_reconfigure/server.h>
#include <rc_control/calibrationConfig.h>
#include <realtime_tools/realtime_buffer.h>

// message
#include <rc_control_msgs/RC_Control.h>
#include <algorithm>

/* 
The RC servo has usual opearting Pulse Width between 1000 and 2000 uS, with default natural positon at 1500 uS.
The Maestro software we use take the Pulse Width in th unit of 0.25 uS.
Therefore, the min-max range is 4000 to 8000 with the natural position at 6000.
*/

class ServoController{
    public:
        // Constructor
        ServoController(ros::NodeHandle node, ros::NodeHandle private_nh);

        // Deconstructor
        ~ServoController();

        /**
        * \brief Updates Servor, send pwd message to both servo and ESC
        * \param time   Current time
        * \param period Time since the last called to update
        */
        void update(const ros::Time& time, const ros::Duration& period);

        void natural();

        void brake();

    // Private Variable
    private:    
        // Maestro Servo Controller
        std::vector<Maestro::Device> _device_list;

        int _device_idx;
        int _steering_ch;
        int _throttle_ch;

        // Ros Node Handler
        ros::NodeHandle _nh;
        ros::NodeHandle _pvt_nh;

        dynamic_reconfigure::Server<rc_control::calibrationConfig> _dyn_reconfig_server;
        dynamic_reconfigure::Server<rc_control::calibrationConfig>::CallbackType _f;
        
        // Servo Controller Status
        bool _running;
        bool _reverse;

        // calibration parameters
        double _steering_C, _steering_L, _steering_R; // center, max left and max right steering input
        double _throttle_N, _throttle_D, _throttle_R; // netural, max forward and max reverse throttle input 
        
        struct DynParam
        {
            double steering_C, steering_L, steering_R; // center, max left and max right steering input
            double throttle_N, throttle_D, throttle_R;
            DynParam() : steering_C(6000), steering_L(4000), steering_R(8000),
                        throttle_N(6000), throttle_D(8000), throttle_R(4000) {}
        };
        realtime_tools::RealtimeBuffer<DynParam> _dynParam;

        // Output limits
        double _steering_min, _steering_max;
        double _throttle_min, _throttle_max;

        // Controller command
        struct Commands
        {
            int throttle;
            int steer;
            bool reverse;
            ros::Time stamp;
            Commands() : throttle(6000), steer(6000), reverse(false), stamp(0.0) {}
        };

        realtime_tools::RealtimeBuffer<Commands> _command;
        std::string _sub_topic;
        Commands _command_struct;

        ros::Subscriber _sub_command;

        /// Timeout to consider cmd_vel commands old:
        double _cmd_vel_timeout;

    //Private Memeber Functions
    private:
        void _ReadParameter();
        int _SearchController();

        // ros callback functions
        void _dynParamCallback(rc_control::calibrationConfig &config, uint32_t level);

        /**
         * \brief controller input subscriber callback
         * \param msg Velocity command message (twist)
         */
        void _subCallback(const rc_control_msgs::RC_Control& msg);

        /**
         * \brief Convert a percentage of input to the target of pwm output
         * \param percentage Input: double (-1 to 1)
         * \return pwm width with unit of 0.25us: int (4000 to 8000)
         */
        int _convertTarget(double percentage, bool is_steer);

        void _dynParamUpdate();

};


#endif