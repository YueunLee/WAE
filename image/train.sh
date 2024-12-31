dataset="CelebA"
data_dir="./data"

model="WAE_GAN"
epochs=100
learning_rate=0.001
adv_learning_rate=0.001
aux_sgd=""

batch_size=64
latent_dim=64
base_channel_size=128

sampler="uniform"
lambda_mmd=200.0
lambda_gan=5.0
penalty_anneal="none"
exact=true

random_seed=2024

python main.py \
--dataset=${dataset} \
--data-dier=${data_dir} \
--model=${model} \
--epochs=${epochs} \
--learning-rate=${learning_rate} \
--adv-learning-rate=${adv_learning_rate} \
--aux-sgd=${aux_sgd} \
--batch-size=${batch_size} \
--latent-dim=${latent_dim} \
--base-channel-size=${base_channel_size} \
--sampler=${sampler} \
--lambda-mmd=${lambda_mmd} \
--lambda-gan=${lambda_gan} \
--penalty-anneal=${penalty_anneal} \
--exact=${exact} \
--random-seed=${random_seed}