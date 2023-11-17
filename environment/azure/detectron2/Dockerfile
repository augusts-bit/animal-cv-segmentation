FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202310.1.v2

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Need to install gdal like this
#RUN python -m pip install --upgrade pip setuptools wheel
#RUN python -m pip install --upgrade gdal
RUN conda install -c conda-forge libgdal
RUN conda install -c conda-forge gdal
RUN conda install krb5
RUN conda install tiledb=2.2
RUN conda install poppler

# Install detectron2 and SAHI
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install SAHI

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh &&     cp /var/configuration/rsyslog.conf /etc/rsyslog.conf &&     cp /var/configuration/nginx.conf /etc/nginx/sites-available/app &&     ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app &&     rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client

