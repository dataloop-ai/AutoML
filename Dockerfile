FROM gcr.io/viewo-g/piper/agent/runner/gpu/main:1.8.0.1

RUN pip3 install --upgrade pip
RUN pip3 install \
	tensorflow==2.0.0 \
	tensorflow-gpu==2.0.0 \
	cffi \
	cython \
	pycocotools \
	opencv-python \
