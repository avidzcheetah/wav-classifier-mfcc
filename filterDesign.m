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

% 4. Feature Extraction Logic

extract_feats = @(x) [
    get_energy(filter(bp1, x)) / (get_energy(filter(bp2, x)) + 1e-6), ...
    get_energy(filter(bp1, x)) / (get_energy(filter(bp3, x)) + 1e-6)
];

% 5. Load Training Set

files_amb = dir(fullfile(train_amb, '*.wav'));
files_fir = dir(fullfile(train_fir, '*.wav'));
X = [];
Y = [];

for i = 1:length(files_amb)
    [audio, ~] = audioread(fullfile(files_amb(i).folder, files_amb(i).name));
    audio = pad_or_trim(audio, segment_len);
    features = extract_feats(audio);
    X = [X; features];
    Y = [Y; "ambulance"];
end

for i = 1:length(files_fir)
    [audio, ~] = audioread(fullfile(files_fir(i).folder, files_fir(i).name));
    audio = pad_or_trim(audio, segment_len);
    features = extract_feats(audio);
    X = [X; features];
    Y = [Y; "firetruck"];
end

% Normalize features
mu = mean(X, 1);
sigma = std(X, 0, 1);
X_norm = (X - mu) ./ sigma;

% 6. Test Phase

test_set = [dir(fullfile(test_amb, '*.wav')); ...
            dir(fullfile(test_fir, '*.wav'))];

res = strings(length(test_set), 3);
hit = 0;

fprintf('\n--- Test Results ---\n');
fprintf('%-25s %-12s %-12s\n', 'File', 'True', 'Predicted');

for i = 1:length(test_set)
    [audio, ~] = audioread(fullfile(test_set(i).folder, test_set(i).name));
    audio = pad_or_trim(audio, segment_len);
    feat = extract_feats(audio);
    feat_norm = (feat - mu) ./ sigma;

    % k-NN (k=3)
    d = vecnorm(X_norm - feat_norm, 2, 2);
    [~, idx] = mink(d, 3);
    nearest = Y(idx);

    counts = tabulate(nearest);
    [~, idx_max] = max(cell2mat(counts(:,2)));
    pred = string(counts{idx_max, 1});

    % Actual label
    if contains(lower(test_set(i).folder), 'ambulance')
        actual = "ambulance";
    else
        actual = "firetruck";
    end

    res(i,:) = [test_set(i).name, actual, pred];

    if actual == pred
        hit = hit + 1;
    end

    fprintf('%-25s %-12s %-12s\n', test_set(i).name, actual, pred);
end

% 7. Accuracy Report

acc = (hit / length(test_set)) * 100;
fprintf('\nClassification Accuracy: %.2f%%\n', acc);

T = table(res(:,1), res(:,2), res(:,3), ...
    'VariableNames', {'FileName', 'ActualClass', 'PredictedClass'});
writetable(T, 'classification_results_part2_unique.csv');

% 8. Helpers

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
