#FROM gcr.io/ris-registry-shared/novnc:ubuntu20.04_cuda11.0
FROM pytorch/pytorch
RUN apt-get update -y && apt-get install -y python3.10
RUN apt-get update -y && apt-get install -y git nano python3-pip binutils libproj-dev gdal-bin ffmpeg libgdal-dev libboost-dev build-essential cmake python3.10-dev libboost-dev 
#python3.9-dev 
#RUN apt-get install -y python3.9-distutils python3.9-venv
#RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.10 1

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 

#RUN update-alternatives --set python3 /usr/bin/python3.10

# Install pip for Python 3.9
#RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
#WORKDIR /home/bmahsa
RUN pip3.10 install napari-PHILOW zarr gunpowder tensorboard tensorflow cython segmentation-models-pytorch
RUN pip3.10 install --upgrade numpy
#RUN git clone https://github.com/funkey/waterz.git
#WORKDIR waterz
#RUN git checkout 8ccd0b308fed604d143577f128420da83ff444da
RUN pip3.10 install git+https://github.com/funkey/waterz@8ccd0b308fed604d143577f128420da83ff444da#egg=waterz 
#--target /usr/local/lib/python3.10/dist-packages
#WORKDIR /home/bmahsa
RUN python3.10 -m pip install --upgrade matplotlib Pillow
RUN pip3.10 install dask

RUN chmod -R 777 /usr/local/lib/python3.10

COPY ./metadata.py /usr/local/lib/python3.9/dist-packages/funlib/persistence/arrays/metadata.py

COPY ./array_source.py /usr/local/lib/python3.9/dist-packages/gunpowder/nodes/array_source.py

RUN chmod -R 777 /usr/local/lib/python3.9/dist-packages