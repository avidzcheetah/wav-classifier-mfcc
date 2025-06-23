% 2022e008, 2022e173
function feat = extract_mfcc_fixed(file, target_len)
    [y, fs] = audioread(file);
    coeffs = mfcc(y, fs);  % Requires Audio Toolbox
    mfcc_mean = mean(coeffs, 1);
    if length(mfcc_mean) < target_len
        feat = [mfcc_mean, zeros(1, target_len - length(mfcc_mean))];
    else
        feat = mfcc_mean(1:target_len);
    end
end
