#!/bin/bash
mkdir -p ISRUC_S3/ExtractedChannels
mkdir -p ISRUC_S3/RawData
echo 'Make data dir: ISRUC_S3'
cd ISRUC_S3/RawData
for s in $(seq 1 10)
do
  wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$s.rar
  unrar x $s.rar
  rm -r $s.rar
done
echo 'Download Data to "ISRUC_S3/RawData" complete.'

cd ISRUC_S3/ExtractedChannels
for s in $(seq 1 100)
do
  wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject$s.mat
done
echo 'Download ExtractedChannels to "ISRUC_S1/ExtractedChannels" complete.'
