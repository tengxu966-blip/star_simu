# Star_simu 仿真环境搭建说明

## 1. Python 环境
建议使用 Python 3.9 及以上版本。

## 2. 安装依赖
推荐使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

如需开发环境支持（如 pytest）：

```bash
pip install -r requirements.txt
pip install pytest
```

## 3. Protocol Buffers
本项目使用 protocol buffers 进行数据结构序列化。

- 安装 protoc 编译器（如未安装）：
  - macOS: `brew install protobuf`
  - Ubuntu: `sudo apt install protobuf-compiler`
- 重新生成 Python 协议文件（如有修改 .proto 文件）：

```bash
protoc --python_out=src src/protocol.proto
```

## 4. 运行仿真

```bash
python run_simulation.py
```

## 5. 生成的图像
仿真过程中生成的图像会自动保存在当前目录下，文件名格式为 `asteroid_时间戳.png`。

## 6. 其他说明
- 如需自定义依赖，请编辑 requirements.txt 或 pyproject.toml。
- 推荐使用虚拟环境（venv、conda）隔离依赖。

如有问题请联系开发者。
