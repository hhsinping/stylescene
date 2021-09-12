mkdir colmap_tat
cd colmap_tat
wget https://storage.googleapis.com/isl-datasets/FreeViewSynthesis/ibr3d_tat.tar.gz
tar zxvf ibr3d_tat.tar.gz
rm ibr3d_tat.tar.gz
wget https://www.dropbox.com/s/dijmv59vnqvs8ad/long.tgz
tar zxvf long.tgz
rm long.tgz
cd ..
mkdir style_data
cd style_data
wget https://3romcg.bn.files.1drv.com/y4m_wXlfBvwva8k9WU4o0ZeFOACIdLr3dN1zxPtGKNmouKJp9jZW4P5K1dDY2T14xl4lAywbFquMxowH4QrgIC5sNnvOMzkviXvMeQqefFxyC05Xaeov3jNW4OfhaFE_-8rtWzyMTuGxuF5uM1XyE_BPzEXg_M7QKE1zn-fqniAOYYnBPj7GeEqQNDf744Wq2C8PJF7ibbc9S6fKWUKoUPtDQ -O style120_data.zip
unzip style120_data.zip
rm style120_data.zip
kaggle competitions download -c painter-by-numbers -f train.zip
unzip train.zip
rm train.zip
cd ..
mkdir stylescene/exp/experiments/
mkdir stylescene/exp/experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/
mkdir stylescene/exp/log
wget https://www.dropbox.com/s/5vw0ll6zeh9djjx/net_0000000000050000.params -O stylescene/exp/experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/net_0000000000050000.params
wget https://www.dropbox.com/s/hqb1k0o57fm6n8g/net_dec.params -O stylescene/exp/projection/net_dec.params
cd preprocess/ext/preprocess/
cmake -DCMAKE_BUILD_TYPE=Release .
make
cd -
cd stylescene/ext/preprocess/
cmake -DCMAKE_BUILD_TYPE=Release .
make
cd -
