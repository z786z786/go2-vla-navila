# Real Go2 Collector

This subtree contains the native real-robot collector and release helpers.

Raw session output must follow `docs/runbooks/raw_session_contract.md` so sim
collectors can match the same upstream schema.

## Build locally

```bash
cmake -S collectors/real_go2/native -B collectors/real_go2/native/build
cmake --build collectors/real_go2/native/build -j
```

## Optional D435i RGB-D capture

The native collector can optionally record a session-level D435i rosbag through
ROS Noetic. When enabled, the collector writes D435i artifacts under:

```text
data/<session_id>/d435i/
  d435i_rgbd.bag
  d435i_capture.json
  rs_camera.launch.log
  rosbag_record.log
```

Example:

```bash
scripts/collect_real_go2.sh \
  --network-interface eno1 \
  --scene-id corridor_a \
  --operator-id op_01 \
  --instruction "go to the door" \
  --d435i
```

Useful flags:

- `--d435i`: auto-launch `realsense2_camera` and record the RGB-D topics
- `--d435i-no-launch`: record only, assuming the ROS camera node is already running
- `--d435i-ros-setup /opt/ros/noetic/setup.bash`
- `--d435i-bag-name d435i_rgbd`
- `--d435i-serial <serial>`
- `--d435i-color-profile 640x480x30`
- `--d435i-depth-profile 640x480x30`
- `--d435i-align-depth`
- `--d435i-launch-file 'realsense2_camera rs_camera.launch'`

When `--d435i-align-depth` is enabled, the collector records the aligned ROS
depth stream and camera info topics:

- `/camera/aligned_depth_to_color/image_raw`
- `/camera/aligned_depth_to_color/camera_info`

When the Web UI is enabled, the collector also exposes a D435i RGB-D preview
panel through collector-owned endpoints:

- `/api/d435i/color.jpg`
- `/api/d435i/depth.jpg`

## Release for upper computer

Use `scripts/release_real_collector.sh` to stage a minimal source release and `scripts/sync_real_collector.sh <orin|pd>` to sync + build on the remote machine.

## Current real-data collection focus

For the current Stage 1.x velocity-policy recovery work, prioritize these
real-robot buckets in order:

1. recovery / correction
2. turning onset / offset
3. start / stop / restart

Avoid spending collection time on generic straight-line cruising unless it is
needed as context around one of the priority buckets above.
