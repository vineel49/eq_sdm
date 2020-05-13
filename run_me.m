% linear equalizer based on STEEPEST DESCENT ALGORITHM
% bpsk modulation, channel impairments: ISI + AWGN
% References: See Section 5.1.2 and 5.1.3 in the book "Digital Communications and
% Signal Processing" by K Vasudevan

clear all
close all
clc
training_len = 10^4;% length of the training sequence
snr_dB = 10; % SNR in dB
equalizer_len = 50; % length of the equalizer
data_len = 10^6; % length of the data sequence
iterations = 100; % number of iterations

% SNR parameters
snr = 10^(0.1*snr_dB);
noise_var = 1/(2*snr); % noise variance
% ---------          training phase       --------------------------------
% source
training_a = randi([0 1],1,training_len);

% bpsk mapper (bit '0' maps to 1 and bit '1' maps to -1)
training_seq = 1-2*training_a;

% isi channel
fade_chan = [0.9 0.1 0.1 -0.1 ]; % impulse response of the ISI channel
fade_chan = fade_chan/norm(fade_chan);
chan_len = length(fade_chan);

% noise
noise = normrnd(0,sqrt(noise_var),1,training_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,training_seq)+noise;

% autocorrelation of the input 
auto_corr_vec = xcorr(chan_op,chan_op,'unbiased');
mid_point = (length(auto_corr_vec)+1)/2;
c = auto_corr_vec(mid_point:mid_point+equalizer_len-1); % first column of toeplitz 
r = fliplr(auto_corr_vec(mid_point-equalizer_len+1:mid_point));
Rvv_Matrix = toeplitz(c,r);

%---------------------------------------------
% cross correlation 
cross_corr_vec = xcorr(training_seq,chan_op(1:length(training_seq)),'unbiased');
MID_POINT = (length(cross_corr_vec)+1)/2;
cross_corr_vec = cross_corr_vec(MID_POINT:MID_POINT+equalizer_len-1).';

%---------------------------------------------
max_step_size = 1/(max(eig(Rvv_Matrix)));% maximum step size
step_size = 0.125*max_step_size;
equalizer = zeros(equalizer_len,1);
for i1= 1:iterations
    equalizer = equalizer+step_size*(cross_corr_vec - Rvv_Matrix*equalizer);
end
equalizer=equalizer.'; % now a row vector

%------------------ data transmission phase----------------------------
% source
data_a = randi([0 1],1,data_len);

% bpsk mapper (bit '0' maps to 1 and bit '1' maps to -1)
data_seq = 1-2*data_a;

% AWGN
noise = normrnd(0,sqrt(noise_var),1,data_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,data_seq)+noise;

% equalization
equalizer_op = conv(chan_op,equalizer);
equalizer_op = equalizer_op(1:data_len);

% MSE (SIMULATION)
MSE_SIMULATION = mean((equalizer_op-data_seq).^2)

% MSE (THEORY)
MSE_THEORY = 1 - cross_corr_vec'*inv(Rvv_Matrix)*cross_corr_vec

% demapping symbols back to bits
dec_a = equalizer_op<0;

% bit error rate
ber = nnz(dec_a-data_a)/data_len