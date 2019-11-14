FROM gcr.io/viewo-g/piper/agent/runner/gpu/main:1.7.53.0

RUN pip3 install --upgrade pip
RUN pip3 install \
	tensorflow==2.0.0 \
	tensorflow-gpu==2.0.0 

