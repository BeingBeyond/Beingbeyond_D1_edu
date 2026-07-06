"""
Example 2: Basic motion test for the head + arm.

This script:
  - Connects to the head + arm controller
  - Moves all joints to the initial pose
  - Then, for each joint in order:
      * Prints the current joint angles and velocities
      * Moves that joint by +15 degrees, then -15 degrees around its initial angle,
        then back to the initial angle
  - Uses wait_until_reached() to block until each small motion is completed

Use this example to:
  - Verify that all head/arm joints are connected and respond
  - Check joint ordering and direction (sign of positive motion)
  - Confirm basic motion without running complex trajectories.
"""
import time
from beingbeyond_d1_edu_sdk import HeadArmRobot

from utils import deg_list_to_rad, rad_list_to_deg


def main():
    print("\033[91mWARNING: Always keep the physical emergency stop button within reach,\033[0m")
    print("\033[91m         and press it immediately if the robot motion looks unsafe.\033[0m\n")

    from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path
    urdf = get_default_urdf_path()
    dev = "/dev/ttyACM0"

    robot = HeadArmRobot(urdf_path=urdf, dev=dev)
    try:
        joint_names = robot.joint_names
        n_joints = len(joint_names)
        print("Joint order:", joint_names)

        # 1. set all joints to the initial position
        q_init_deg = [0.0, 0.0,
                      0.0, -60.0, 60.0, 0.0, 0.0, 0.0]
        motion_range_deg = 15.0
        wait_pos_tol_deg = 10.0
        wait_vel_tol_deg_s = 20.0
        wait_timeout_s = 1.0
        if len(q_init_deg) != n_joints:
            raise ValueError(
                f"q_init_deg length ({len(q_init_deg)}) does not match "
                f"robot joint count ({n_joints})"
            )

        q_init_rad = deg_list_to_rad(q_init_deg)
        print(f"\n[Step] Move all joints to initial pose (deg): {q_init_deg}")
        robot.set_positions(q_init_rad)
        # blocking wait until reached
        robot.wait_until_reached(
            q_init_rad,
            active_joint_indices=range(n_joints),
            pos_tol_deg=wait_pos_tol_deg,
            vel_tol_deg_s=wait_vel_tol_deg_s,
            timeout_s=3.0,
        )
        time.sleep(0.5)

        def wait_for_joint(target_rad, joint_idx):
            return robot.wait_until_reached(
                target_rad,
                active_joint_indices=[joint_idx],
                pos_tol_deg=wait_pos_tol_deg,
                vel_tol_deg_s=wait_vel_tol_deg_s,
                timeout_s=wait_timeout_s,
            )

        # 2. for each joint, move around q_init_deg, then return to q_init_deg
        for idx, name in enumerate(joint_names):
            print(f"\n====== Joint {idx}: {name} ======")

            # get current state
            q_rad, dq_rad = robot.get_positions_and_velocities()
            q_deg = rad_list_to_deg(q_rad)
            dq_deg = rad_list_to_deg(dq_rad)

            print(f"  Current q (deg): {q_deg}")
            print(f"  Current dq (deg/s) approx: {dq_deg}")
            print(f"  This joint q[{idx}] = {q_deg[idx]:.2f} deg, dq[{idx}] = {dq_deg[idx]:.2f} deg/s")


            base_deg = q_init_deg.copy()
            base_joint_deg = base_deg[idx]

            # 2.1 q_init + motion_range_deg
            target_deg = base_deg.copy()
            target_deg[idx] = base_joint_deg + motion_range_deg
            target_rad = deg_list_to_rad(target_deg)
            print(
                f"  Move joint {idx} ({name}) to "
                f"{target_deg[idx]:.2f} deg (q_init + {motion_range_deg:.2f})"
            )
            robot.set_positions(target_rad)
            wait_for_joint(target_rad, idx)

            # 2.2 q_init - motion_range_deg
            target_deg = base_deg.copy()
            target_deg[idx] = base_joint_deg - motion_range_deg
            target_rad = deg_list_to_rad(target_deg)
            print(
                f"  Move joint {idx} ({name}) to "
                f"{target_deg[idx]:.2f} deg (q_init - {motion_range_deg:.2f})"
            )
            robot.set_positions(target_rad)
            wait_for_joint(target_rad, idx)

            # 2.3 back to q_init_deg
            target_deg = base_deg.copy()
            target_rad = deg_list_to_rad(target_deg)
            print(f"  Move joint {idx} ({name}) back to {base_joint_deg:.2f} deg")
            robot.set_positions(target_rad)
            wait_for_joint(target_rad, idx)

        print(
            "\n[Done] All joints tested "
            f"(q_init +/- {motion_range_deg:.2f} deg, then back to q_init)."
        )
    finally:
        robot.home_and_close()

if __name__ == "__main__":
    main()
