import rospy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from traj_tracking_dyn import TrajTrackingDyn
from traj_tracking_kin import TrajTrackingKin

class MPC:
    def __init__(self) -> None:
        
