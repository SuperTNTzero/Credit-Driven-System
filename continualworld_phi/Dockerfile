# syntax=docker/dockerfile:1.7
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl
ENV LD_LIBRARY_PATH=/opt/.mujoco/mujoco200/bin
ENV MUJOCO_PY_MUJOCO_PATH=/opt/.mujoco/mujoco200
ENV MUJOCO_PY_MJKEY_PATH=/opt/.mujoco/mjkey.txt
ENV PYTHONPATH=/workspace/continual_world:/workspace/project

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget curl unzip \
    build-essential patchelf libgl1-mesa-dev libgl1-mesa-glx \
    libosmesa6-dev libglew-dev libglfw3 libglfw3-dev \
    libxrender1 libxext6 libsm6 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pin the upstream implementation in the container rather than installing
# it into the host environment. The repository itself remains mounted/copyable.
RUN git clone --depth 1 https://github.com/awarelab/continual_world.git /workspace/continual_world
RUN python3 -m pip install \
    "pip==23.0.1" "setuptools==65.5.0" "wheel==0.38.4"
RUN mkdir -p /opt/.mujoco
#RUN rm -rf /root/.mujoco/mujoco-py /tmp/pip-install-* || true
# The legacy benchmark depends on MuJoCo 2.0 and mujoco-py. Supply the license
# as a BuildKit secret during installation; never persist it in an image layer.
RUN wget -q https://www.roboti.us/download/mujoco200_linux.zip -O /tmp/mujoco.zip \
    && unzip -q /tmp/mujoco.zip -d /opt/.mujoco \
    && mv /opt/.mujoco/mujoco200_linux /opt/.mujoco/mujoco200 \
    && rm /tmp/mujoco.zip

RUN pip3 install \
    "Cython==0.29.37" \
    "fasteners==0.19" \
    "glfw==2.6.5" \
    "imageio==2.34.2" \
    "cffi==1.17.1"

RUN --mount=type=secret,id=mjkey,target=/opt/.mujoco/mjkey.txt,required=true \
    test -s /opt/.mujoco/mjkey.txt && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip3 config set global.timeout 600 && \
    pip3 install --no-cache-dir \
    "numpy==1.23.5" \
    "tensorflow==2.13.*" \
    pandas matplotlib seaborn gym==0.21.0 && \
    pip3 install --no-build-isolation --no-cache-dir "mujoco-py==2.0.2.13" && \
    python3 -m pip install -e /workspace/continual_world && \
    pip3 install --no-deps "Cython==0.29.37" && \
    python3 -c "import Cython; assert Cython.__version__ == '0.29.37', Cython.__version__; import mujoco_py; import continualworld"

COPY continualworld_phi /workspace/project/continualworld_phi
WORKDIR /workspace/project

CMD ["bash"]
