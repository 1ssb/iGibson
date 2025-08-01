#!/usr/bin/env python3
"""
iGibson Interactive Controller - Minimal Version
WASD movement with data saving
"""

import os
import sys
import numpy as np
import cv2
import time
from omegaconf import OmegaConf
import json

sys.path.insert(0, '~/src/iGibson')

import igibson
from igibson.envs.igibson_env import iGibsonEnv

CONFIG_PATH = 'data/configs/ig_config.yaml'
MOVE_SPEED = 0.5
TURN_SPEED = 0.5

class iGibsonController:
    def __init__(self, scene_name=None):
        self.config = OmegaConf.load(CONFIG_PATH)
        if scene_name:
            self.config.scene_id = scene_name
        self.scene_name = self.config.scene_id
        self.env = None
        self.current_obs = None
        self.data_dir = f'data_ac/{self.scene_name}'
        os.makedirs(self.data_dir, exist_ok=True)
        # Load instance ID to label mapping from scene JSON
        self.instance_id_to_label = self.load_instance_mapping()

    def load_instance_mapping(self):
        scenes_base = os.path.join(igibson.ig_dataset_path, 'scenes')
        scene_dir = os.path.join(scenes_base, self.scene_name)
        possible_jsons = [
            'scene_instances.json',
            'semantic_class.json',
            'object_semantics.json',
        ]
        for json_name in possible_jsons:
            json_path = os.path.join(scene_dir, json_name)
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        mapping = json.load(f)
                    print(f"Loaded instance mapping from {json_path}")
                    return mapping
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
        print("No instance mapping JSON found for this scene.")
        return {}

    def initialize_environment(self):
        try:
            config = OmegaConf.to_container(self.config, resolve=True)
            config['task'] = 'point_nav_random'
            config['reward_type'] = 'geodesic'
            config['success_reward'] = 0.0
            config['potential_reward_weight'] = 0.0
            config['collision_reward_weight'] = 0.0
            config['max_step'] = 999999
            config['max_collisions_allowed'] = 999999
            config['visible_target'] = False
            config['visible_path'] = False
            config['load_object_categories'] = []
            config['enable_shadow'] = True
            config['enable_pbr'] = True

            print("Initializing environment with config:", {k: v for k, v in config.items() if k in ['scene_id', 'mode', 'task', 'reward_type']})

            self.env = iGibsonEnv(config_file=config, mode="gui_interactive")
            self.current_obs = self.env.reset()
            print("Environment initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing environment: {e}")
            import traceback
            traceback.print_exc()
            return False

    def display_sensor_data(self):
        obs = self.current_obs
        if obs is None:
            return

        # Ensure OpenCV window exists for keyboard events
        cv2.namedWindow('RGB Camera View', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth Camera View', cv2.WINDOW_AUTOSIZE)

        if 'rgb' in obs:
            rgb = (obs['rgb'] * 255).astype(np.uint8) if obs['rgb'].max() <= 1.0 else obs['rgb'].astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(rgb_bgr, "WASD: Move | C: Capture | X: Exit", (10, rgb_bgr.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('RGB Camera View', rgb_bgr)

        if 'depth' in obs:
            depth = obs['depth']
            depth_vis = depth[:, :, 0] if len(depth.shape) == 3 and depth.shape[2] == 1 else depth
            if depth_vis.max() > depth_vis.min():
                depth_norm = ((depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min()) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_vis, dtype=np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            cv2.imshow('Depth Camera View', depth_colored)

    def save_current_data(self):
        obs = self.current_obs
        if obs is None:
            print("Obs is None")
            return

        timestamp = int(time.time() * 1000)
        # Save RGB
        if 'rgb' in obs:
            rgb = (obs['rgb'] * 255).astype(np.uint8) if obs['rgb'].max() <= 1.0 else obs['rgb'].astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb_path = os.path.join(self.data_dir, f'{timestamp}_RGB.png')
            ok = cv2.imwrite(rgb_path, rgb_bgr)
            if ok:
                print(f"RGB image saved: {rgb_path}")
            else:
                print(f"ERROR: Failed to save RGB image: {rgb_path}")

        # Save Depth
        if 'depth' in obs:
            depth_path = os.path.join(self.data_dir, f'{timestamp}_depth.npy')
            try:
                np.save(depth_path, obs['depth'])
                if os.path.exists(depth_path) and os.path.getsize(depth_path) > 0:
                    print(f"Depth array saved: {depth_path}")
                else:
                    print(f"ERROR: Depth array not saved properly: {depth_path}")
            except Exception as e:
                print(f"ERROR: Failed to save depth array: {e}")

    def handle_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF
        linear_vel, angular_vel = 0.0, 0.0

        if key == ord('w'): linear_vel = MOVE_SPEED
        elif key == ord('s'): linear_vel = -MOVE_SPEED
        elif key == ord('d'): angular_vel = TURN_SPEED
        elif key == ord('a'): angular_vel = -TURN_SPEED
        elif key == ord('c'): self.save_current_data()
        elif key == ord('x'): return False, linear_vel, angular_vel

        return True, linear_vel, angular_vel

    def run_interactive_loop(self):
        if not self.initialize_environment():
            return

        print(f"Controls: WASD=Move, C=Capture, X=Exit | Data: {self.data_dir}")
        print("Robot should be visible in the 3D window. Use WASD to move around.")

        try:
            while True:
                cont, linear_vel, angular_vel = self.handle_keyboard_input()
                if not cont:
                    break

                action = np.array([linear_vel, angular_vel])
                self.current_obs, reward, done, info = self.env.step(action)

                # No more continuous RGB/depth debug prints!
                self.display_sensor_data()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        if self.env:
            self.env.close()
        cv2.destroyAllWindows()

def main():
    scenes_path = os.path.join(igibson.ig_dataset_path, 'scenes')
    available_scenes = [scene for scene in os.listdir(scenes_path)
                        if not scene.startswith('background') and os.path.isdir(os.path.join(scenes_path, scene))]

    print("Available scenes:")
    for i, scene in enumerate(available_scenes):
        print(f"  {i+1}. {scene}")

    try:
        scene_num = int(input("Which scene? (1-15): ")) - 1
    except:
        scene_num = 0

    if scene_num < 0 or scene_num >= len(available_scenes):
        scene_num = 0

    controller = iGibsonController(available_scenes[scene_num])
    controller.run_interactive_loop()

if __name__ == "__main__":
    main()
