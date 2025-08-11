#!/bin/bash
# Simple script to create a GCP GPU instance for ML experiments

set -e

# Configuration
INSTANCE_NAME="ml-experiments"
ZONE="us-central1-a"
MACHINE_TYPE="g2-standard-4"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="100GB"

echo "Creating GCP GPU instance..."
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Machine: $MACHINE_TYPE"
echo ""

# Create the instance
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --metadata=startup-script='#!/bin/bash
    # Download and run setup script
    cd /home/$(ls /home | head -1)
    wget -O setup.sh https://raw.githubusercontent.com/cemoody/arcagi/main/setup.sh
    chmod +x setup.sh
    echo "Setup script downloaded. Run: ./setup.sh"
    '

echo ""
echo "Instance created successfully!"
echo ""
echo "To connect:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "After connecting, run: ./setup.sh"
echo ""
echo "Instance will auto-terminate as a spot instance or after 24 hours max"