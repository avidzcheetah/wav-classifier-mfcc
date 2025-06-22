clc;
clear;

% 1. Setup and folder paths
train_amb = 'train/ambulance';
train_fir = 'train/firetruck';
test_amb = 'test/ambulance';
test_fir = 'test/firetruck';
segment_len = 16384;

% 2. Get sampling rate
samp_file = dir(fullfile(train_amb, '*.wav'));
[y_temp, fs] = audioread(fullfile(samp_file(1).folder, samp_file(1).name));

% Plot spectrum of first ambulance and firetruck file
files_amb_plot = dir(fullfile(train_amb, '*.wav'));
files_fir_plot = dir(fullfile(train_fir, '*.wav'));
[y_amb, ~] = audioread(fullfile(files_amb_plot(1).folder, files_amb_plot(1).name));
[y_fir, ~] = audioread(fullfile(files_fir_plot(1).folder, files_fir_plot(1).name));
y_amb = y_amb(:,1); y_fir = y_fir(:,1);
N = min(length(y_amb), length(y_fir));
y_amb = y_amb(1:N); y_fir = y_fir(1:N);
f = (0:N-1)*(fs/N);
Y_amb = abs(fft(y_amb));
Y_fir = abs(fft(y_fir));

figure('Name','Frequency Spectra');
subplot(2,1,1); plot(f(1:N/2), Y_amb(1:N/2));
title('Ambulance Spectrum'); xlabel('Frequency (Hz)'); ylabel('Magnitude'); grid on;
subplot(2,1,2); plot(f(1:N/2), Y_fir(1:N/2));
title('Firetruck Spectrum'); xlabel('Frequency (Hz)'); ylabel('Magnitude'); grid on;

% 3. Filter Design
bp1 = designfilt('bandpassiir', 'FilterOrder', 6, ...
    'HalfPowerFrequency1', 600, 'HalfPowerFrequency2', 1100, ...
    'SampleRate', fs);
bp2 = designfilt('bandpassiir', 'FilterOrder', 6, ...
    'HalfPowerFrequency1', 1300, 'HalfPowerFrequency2', 1800, ...
    'SampleRate', fs);
bp3 = designfilt('bandpassiir', 'FilterOrder', 6, ...
    'HalfPowerFrequency1', 2100, 'HalfPowerFrequency2', 2800, ...
    'SampleRate', fs);

% Plot filter frequency responses
figure('Name','Filter Responses');
[h1,f1] = freqz(bp1, fs); [h2,f2] = freqz(bp2, fs); [h3,f3] = freqz(bp3, fs);
plot(f1, 20*log10(abs(h1)), 'r', f2, 20*log10(abs(h2)), 'g', f3, 20*log10(abs(h3)), 'b');
legend('600-1100 Hz','1300-1800 Hz','2100-2800 Hz'); title('Bandpass Filter Responses');
xlabel('Frequency (Hz)'); ylabel('Gain (dB)'); grid on;

% 4. Feature Extraction Logic
extract_feats = @(x) [
    get_energy(filter(bp1, x)) / (get_energy(filter(bp2, x)) + 1e-6), ...
    get_energy(filter(bp1, x)) / (get_energy(filter(bp3, x)) + 1e-6)
];

% 5. Load Training Set
files_amb = dir(fullfile(train_amb, '*.wav'));
files_fir = dir(fullfile(train_fir, '*.wav'));
X = []; Y = [];

for i = 1:length(files_amb)
    [audio, ~] = audioread(fullfile(files_amb(i).folder, files_amb(i).name));
    audio = pad_or_trim(audio, segment_len);
    features = extract_feats(audio);
    X = [X; features]; Y = [Y; "ambulance"];
end

for i = 1:length(files_fir)
    [audio, ~] = audioread(fullfile(files_fir(i).folder, files_fir(i).name));
    audio = pad_or_trim(audio, segment_len);
    features = extract_feats(audio);
    X = [X; features]; Y = [Y; "firetruck"];
end

% Normalize features
mu = mean(X, 1); sigma = std(X, 0, 1);
X_norm = (X - mu) ./ sigma;

% Plot feature scatter
figure('Name','Training Feature Scatter');
scatter(X(Y=="ambulance",1), X(Y=="ambulance",2), 'b', 'filled'); hold on;
scatter(X(Y=="firetruck",1), X(Y=="firetruck",2), 'r', 'filled');
xlabel('E1 / E2'); ylabel('E1 / E3'); legend('Ambulance','Firetruck');
title('Feature Ratios of Training Data'); grid on; hold off;

% 6. Test Phase
test_set = [dir(fullfile(test_amb, '*.wav')); dir(fullfile(test_fir, '*.wav'))];
res = strings(length(test_set), 3); hit = 0;

fprintf('\n--- Test Results ---\n');
fprintf('%-25s %-12s %-12s\n', 'File', 'True', 'Predicted');

for i = 1:length(test_set)
    [audio, ~] = audioread(fullfile(test_set(i).folder, test_set(i).name));
    audio = pad_or_trim(audio, segment_len);
    feat = extract_feats(audio);
    feat = feat(:)';  % Ensure row vector
    feat_norm = (feat - mu) ./ sigma;

    d = vecnorm(X_norm - feat_norm, 2, 2);
    [~, idx] = mink(d, 3);
    nearest = Y(idx);
    counts = tabulate(nearest);
    [~, idx_max] = max(cell2mat(counts(:,2)));
    pred = string(counts{idx_max, 1});

    if contains(lower(test_set(i).folder), 'ambulance')
        actual = "ambulance";
    else
        actual = "firetruck";
    end

    res(i,:) = [test_set(i).name, actual, pred];
    if actual == pred, hit = hit + 1; end

    fprintf('%-25s %-12s %-12s\n', test_set(i).name, actual, pred);
end

% 7. Accuracy Report
acc = (hit / length(test_set)) * 100;
fprintf('\nClassification Accuracy: %.2f%%\n', acc);
T = table(res(:,1), res(:,2), res(:,3), ...
    'VariableNames', {'FileName', 'ActualClass', 'PredictedClass'});
writetable(T, 'classification_results_part2_unique.csv');

% 8. Filtered Signal Plot
y_sample = y_amb;  % Use first ambulance signal
y_f1 = filter(bp1, y_sample);
y_f2 = filter(bp2, y_sample);
y_f3 = filter(bp3, y_sample);

figure('Name','Filtered Signal Comparison');
subplot(4,1,1); plot(y_sample); title('Original Signal');
xlabel('Samples'); ylabel('Amplitude'); grid on;
subplot(4,1,2); plot(y_f1); title('Band 600-1100 Hz'); grid on;
subplot(4,1,3); plot(y_f2); title('Band 1300-1800 Hz'); grid on;
subplot(4,1,4); plot(y_f3); title('Band 2100-2800 Hz'); grid on;

% 9. Helpers
function e = get_energy(sig)
    e = sum(sig.^2);
end

function out = pad_or_trim(sig, len)
    sig = sig(:,1); % mono
    if length(sig) < len
        out = [sig; zeros(len - length(sig), 1)];
    else
        out = sig(1:len);
    end
end
