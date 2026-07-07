# 中文版 README

## 简介

本项目用于将 **Apple Vision Pro** 的手部追踪数据映射到 **Isaac Gym** 中的 D1 机器人，支持两种使用方式：

- `d1_isaac_control.py`：实时遥操作
- `d1_isaac_replay.py`：离线回放

当前目录结构示例：

```text
avp_teleop/
├── d1_isaac_control.py
├── d1_isaac_replay.py
├── demo_data/
├── dex-retargeting/
├── README.md
└── README_中文.md
```

---

## 1. 环境要求

- Python **3.8**（建议）
- Isaac Gym (**Preview 4**)
- 已安装并可导入：
  - `beingbeyond_d1_sdk`
  - `avp_stream`
  - `pinocchio`
  - `scipy`
  - `opencv-python`

### 重要版本（建议固定）
- `numpy==1.24.4`
- `pin==2.7.0`


---

## 2. 安装步骤


### 步骤 1：进入项目目录
```bash
cd avp_teleop
```

### 步骤 2：安装关键版本依赖
```bash
pip install "avp_stream"
pip install "numpy==1.24.4"
pip install "pin==2.7.0"
```

### 步骤 3：安装本地 `dex_retargeting`

```bash
cd dex-retargeting
pip install -e .
cd ..
```

安装后可检查：
```bash
pip show dex_retargeting
```

### 步骤 4：安装其它常用依赖
```bash
pip install scipy opencv-python
```

---

## 3. 运行

### A. 实时遥操作（Vision Pro）

脚本：`d1_isaac_control.py`

功能：
- 子进程连接 Vision Pro 并写入共享内存
- 主进程读取共享内存，执行手部重定向 + IK，并驱动 Isaac Gym
- 可选将 Isaac Gym 相机画面回传到 Vision Pro

运行命令：
```bash
python d1_isaac_control.py --ip <VISION_PRO_IP>
```

示例：
```bash
python d1_isaac_control.py --ip 192.168.20.100
```

说明：
- 不传 `--ip` 时，会以纯仿真模式运行（无外部输入）：
  ```bash
  python d1_isaac_control.py
  ```
- 打开 Isaac Gym viewer 后，按 **`S`**：
  - 执行标定（记录当前手部为参考原点）
  - 开始遥操作

---

### B. 离线回放（`.npz` 数据）

脚本：`d1_isaac_replay.py`

运行命令：
```bash
python d1_isaac_replay.py
```

说明：
- 默认读取脚本中 `ReplayCfg.tracking_data_path` 指定的 `.npz`
- 回放结束后会自动重置并循环播放

如果你要使用自己的数据文件，请修改 `d1_isaac_replay.py` 中的：
```python
tracking_data_path = "demo_data/demo_data.npz"
```

---

## 4. `.npz` 数据格式（回放用）

`d1_isaac_replay.py` 需要的字段：

### 必需字段
- `timestamps`: `(T,)`
- `right_wrist_poses`: `(T, 4, 4)`
- `right_pinch`: `(T,)`

### 可选字段
- `right_fingers`: `(T, 25, 4, 4)`

