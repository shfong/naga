#!/usr/bin/env bash

# install base packages
yum install -y epel-release git gzip tar
yum install -y wget bzip2 bzip2-utils bzip2-devel gcc gcc-c++ hdf5 hdf5-devel 
yum install -y lzo lzo-devel libxml2-devel

# install miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh

# install miniconda
./Miniconda3-latest-Linux-x86_64.sh -p /opt/miniconda3 -b
rm ./Miniconda3-latest-Linux-x86_64.sh

# set path to miniconda -- should really add to /etc/profile.d so everyone gets it
export PATH=/opt/miniconda3/bin:$PATH
echo "export PATH=/opt/miniconda3/bin:$PATH" >> /root/.bash_profile
echo "export PATH=/opt/miniconda3/bin:$PATH" >> /root/.bashrc
sudo -u vagrant echo "export PATH=/opt/miniconda3/bin:$PATH" >> /home/vagrant/.bash_profile

# install ndex2-client
git clone -b 'chrisdev' --single-branch --depth 1 https://github.com/ndexbio/ndex2-client.git
pushd ndex2-client
python setup.py bdist_wheel
pip install dist/ndex2-*whl
popd
rm -rf ndex2-client

# install nbgwas
git clone https://github.com/shfong/nbgwas.git
pushd nbgwas
python setup.py bdist_wheel
pip install dist/nbgwas*whl
popd
rm -rf nbgwas
