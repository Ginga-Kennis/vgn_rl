import rospy
import geometry_msgs.msg
import numpy as np
from std_srvs.srv import SetBool, Empty

from robot_helpers.ros import tf
from robot_helpers.ros.robotiq_gripper import RobotiqGripperClient
from robot_helpers.ros.moveit import MoveItClient
from robot_helpers.spatial import Rotation, Transform
from vgn.rviz import Visualizer
from vgn.srv import GetMapCloud, GetSceneCloud, PredictGrasps, PredictGraspsRequest
from vgn.utils import from_grasp_config_msg

class Ur5eGraspController(object):
    def __init__(self):
        self.load_parameters()
        self.lookup_transforms()
        self.init_robot_connection()
        self.init_services()
        self.vis = Visualizer()
        rospy.sleep(1.0)
        rospy.loginfo("Ready to take action")
        
        
    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.ee_frame = rospy.get_param("~ee_frame_id")
        self.task_frame = rospy.get_param("~task_frame_id")
        self.ee_grasp_offset = Transform.from_list(rospy.get_param("~ee_grasp_offset"))
        self.grasp_ee_offset = self.ee_grasp_offset.inv()
        self.scan_joints = rospy.get_param("~scan_joints")
        self.place_joints = rospy.get_param("~place_joints")

    def lookup_transforms(self):
        tf.init()
        self.T_base_task = tf.lookup(self.base_frame, self.task_frame)

    def init_robot_connection(self):
        self.gripper = RobotiqGripperClient()
        self.moveit = MoveItClient("ur5e")
        self.moveit.move_group.set_end_effector_link(self.ee_frame)

        # Add a box to the planning scene to avoid collisions with the table.
        # To Do (change table position)
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.task_frame
        msg.pose.position.x = 0.10
        msg.pose.position.y = 0.10
        msg.pose.position.z = -0.02
        self.moveit.scene.add_box("table", msg, size=(1.0, 0.6, 0.02))

    def init_services(self):
        self.reset_map = rospy.ServiceProxy("reset_map", Empty)
        self.toggle_integration = rospy.ServiceProxy("toggle_integration", SetBool)
        self.get_scene_cloud = rospy.ServiceProxy("get_scene_cloud", GetSceneCloud)
        self.get_map_cloud = rospy.ServiceProxy("get_map_cloud", GetMapCloud)
        self.predict_grasps = rospy.ServiceProxy("predict_grasps", PredictGrasps)

    def run(self):
        self.vis.clear()
        self.gripper.open()

        self.vis.roi(self.task_frame, 0.3)

        rospy.loginfo("Reconstructing scene")
        self.scan_scene()
        self.get_scene_cloud()       # Pointcloud
        res = self.get_map_cloud()   # TSDF

        rospy.loginfo("Planning grasps")
        req = PredictGraspsRequest(res.voxel_size, res.map_cloud)
        res = self.predict_grasps(req)

        if len(res.grasps) == 0:
            rospy.loginfo("No grasps detected")
            return
        
        # Deserialize result
        grasps = []
        for msg in res.grasps:
            grasp, _ = from_grasp_config_msg(msg)
            grasps.append(grasp)

        # Select the highest grasp
        scores = [grasp.pose.translation[2] for grasp in grasps]
        grasp = grasps[np.argmax(scores)]

        # Execute grasp
        rospy.loginfo("Executing grasp")
        self.moveit.check_and_goto("ready")
        self.execute_grasp(grasp)

        
        rospy.loginfo("Dropping object")
        self.moveit.check_and_goto(self.place_joints[0])
        self.gripper.open()
        

    def execute_grasp(self, grasp):
        # Transform to base frame
        grasp.pose = self.T_base_task * grasp.pose

        # Ensure that the camera is pointing forward.
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        T_base_grasp = grasp.pose

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.10])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.20])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.moveit.check_and_goto(T_base_pregrasp * self.grasp_ee_offset, velocity_scaling=0.2)
        self.moveit.gotoL(T_base_grasp * self.grasp_ee_offset)  # Linear movement
        self.gripper.close()
        self.moveit.gotoL(T_base_retreat * self.grasp_ee_offset)


    def scan_scene(self):
        self.moveit.check_and_goto("ready")
        self.reset_map()
        self.toggle_integration(True)
        for joint_target in self.scan_joints:
            self.moveit.check_and_goto(joint_target)
        self.toggle_integration(False)


if __name__ == "__main__":
    rospy.init_node("ur5e_grasp")
    controller = Ur5eGraspController()
    while True:
        controller.run()