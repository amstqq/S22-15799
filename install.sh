sudo apt-get install zip unzip
yes | pip install -r requirements.txt

# Install Hypopg - https://hypopg.readthedocs.io/en/rel1_stable/installation.html
sudo apt -y install postgresql-14-hypopg


# Install dexter - https://github.com/ankane/dexter/blob/master/guides/Linux.md
wget -qO- https://dl.packager.io/srv/pghero/dexter/key | sudo apt-key add -
sudo wget -O /etc/apt/sources.list.d/dexter.list \
  https://dl.packager.io/srv/pghero/dexter/master/installer/ubuntu/$(. /etc/os-release && echo $VERSION_ID).repo
sudo apt-get update
sudo apt-get -y install dexter