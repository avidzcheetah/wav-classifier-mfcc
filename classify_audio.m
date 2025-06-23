% 2022e008, 2022e173
clear; clc;

% Folder path
base_path = 'C:\Users\Chetana\Desktop\task2\Part 01';
class1_path = fullfile(base_path, 'class_1');
class2_path = fullfile(base_path, 'class_2');
unknown_path = fullfile(base_path, 'unknown');

% List .wav files
class1_files = dir(fullfile(class1_path, '*.wav'));
class2_files = dir(fullfile(class2_path, '*.wav'));
unknown_files = dir(fullfile(unknown_path, '*.wav'));

% Feature vector length
target_feature_length = 13;

fprintf('Extracting features...\n');

%% ---------- CLASS 1 ----------
features_class1 = zeros(length(class1_files), target_feature_length);
figure('Name', 'Class 1 Waveforms'); 
for k = 1:length(class1_files)
    fpath = fullfile(class1_path, class1_files(k).name);
    features_class1(k, :) = extract_mfcc_fixed(fpath, target_feature_length);

    % 📈 Plot waveform as subplot
    [y, fs] = audioread(fpath);
    subplot(ceil(length(class1_files)/2), 2, k);  % 2 columns
    plot((1:length(y))/fs, y);
    title(['class_1: ', class1_files(k).name], 'Interpreter', 'none');
    xlabel('Time (s)'); ylabel('Amplitude');
end

%% ---------- CLASS 2 ----------
features_class2 = zeros(length(class2_files), target_feature_length);
figure('Name', 'Class 2 Waveforms'); 
for k = 1:length(class2_files)
    fpath = fullfile(class2_path, class2_files(k).name);
    features_class2(k, :) = extract_mfcc_fixed(fpath, target_feature_length);

    % 📈 Plot waveform as subplot
    [y, fs] = audioread(fpath);
    subplot(ceil(length(class2_files)/2), 2, k);  % 2 columns
    plot((1:length(y))/fs, y);
    title(['class_2: ', class2_files(k).name], 'Interpreter', 'none');
    xlabel('Time (s)'); ylabel('Amplitude');
end

%% ---------- UNKNOWN ----------
fprintf('Classifying unknown files...\n');
results = {};
figure('Name', 'Unknown Waveforms'); 
for k = 1:length(unknown_files)
    fname = unknown_files(k).name;
    fpath = fullfile(unknown_path, fname);
    feat = extract_mfcc_fixed(fpath, target_feature_length);

    d1 = mean(vecnorm(features_class1 - feat, 2, 2));
    d2 = mean(vecnorm(features_class2 - feat, 2, 2));

    if d1 < d2
        predicted_class = 'class_1';
    else
        predicted_class = 'class_2';
    end

    results{end+1, 1} = fname;
    results{end, 2} = predicted_class;

    % 📈 Plot waveform as subplot
    [y, fs] = audioread(fpath);
    subplot(ceil(length(unknown_files)/2), 2, k);  % 2 columns
    plot((1:length(y))/fs, y);
    title(['unknown: ', fname], 'Interpreter', 'none');
    xlabel('Time (s)'); ylabel('Amplitude');
end

%% ---------- RESULTS ----------
fprintf('\nClassification Results:\n');
fprintf('%-20s | %-10s\n', 'Filename', 'Predicted');
fprintf('----------------------|------------\n');
for i = 1:size(results, 1)
    fprintf('%-20s | %-10s\n', results{i,1}, results{i,2});
end

% Save to CSV
cell2csv('part1_results.csv', [{'Filename', 'Predicted Class'}; results]);
fprintf('\n✅ Results saved to part1_results.csv\n');
