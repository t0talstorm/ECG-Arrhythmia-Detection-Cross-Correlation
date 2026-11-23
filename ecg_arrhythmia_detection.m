%% ECG ARRHYTHMIA DETECTION USING CROSS-CORRELATION (TEMPLATE MATCHING)
% MIT-BIH Arrhythmia Database - Record 200 (PhysioNet) | Fs = 360 Hz
% Requires: '100.csv' exported from PhysioNet in the working directory.
%
% CSV download (Method B - recommended):
%   pip install wfdb pandas
%   import wfdb, pandas as pd
%   r = wfdb.rdrecord('200', pn_dir='mitdb', sampfrom=0, sampto=3600)
%   df = pd.DataFrame(r.p_signal, columns=r.sig_name)
%   df.insert(0, 'time_s', [i/360.0 for i in range(len(df))])
%   df.to_csv('200.csv', index=False)

clc; clear; close all;

%% STEP 1 : SYSTEM PARAMETERS


csv_filename  = '100.csv';
[~, record_name, ~] = fileparts(csv_filename);

Fs            = 360;     % [Hz] MIT-BIH sampling frequency
HP_CUTOFF     = 0.5;     % [Hz] High-pass cutoff for baseline wander removal
HP_ORDER      = 4;       % Butterworth filter order
TEMPLATE_HALF = 40;      % Half-window for template (+/-40 samples = ~0.222 s)
SEARCH_START  = 0.5;     % [s] Search window start for Normal beat extraction
SEARCH_END    = 1.5;     % [s] Search window end
CORR_THRESH   = 0.70;    % Correlation threshold: >= Normal, < Arrhythmia
MIN_BEAT_SEP  = 0.25;    % [s] Minimum inter-beat interval
FIGURE_DPI    = 300;

%% STEP 2 : FILE CHECK & DATA LOADING

if ~isfile(csv_filename)
    error('File not found: "%s". See script header for download instructions.', csv_filename);
end

fprintf('Loading ECG from "%s"...\n', csv_filename);

fid = fopen(csv_filename, 'r');
if fid == -1
    error('Cannot open file: %s', csv_filename);
end

fgetl(fid);   % Discard header line
raw_lines = textscan(fid, '%s %f %f', 'Delimiter', ',', 'CollectOutput', false);
fclose(fid);

ecg_raw = raw_lines{2};   % MLII channel [mV]
N       = length(ecg_raw);

% Use a clean uniform time axis (more reliable than parsing the string time column)
time_raw = (0 : N-1)' / Fs;

fprintf('Loaded %d samples | Duration = %.3f s | Fs = %d Hz\n\n', N, time_raw(end), Fs);

%% STEP 3 : PRE-PROCESSING — BASELINE WANDER REMOVAL
% 4th-order zero-phase Butterworth HPF (filtfilt) removes respiratory drift
% below 0.5 Hz without introducing group delay that would shift R-peak positions.

Wn = HP_CUTOFF / (Fs/2);
[b_hp, a_hp] = butter(HP_ORDER, Wn, 'high');
ecg_filtered = filtfilt(b_hp, a_hp, ecg_raw);

fprintf('Baseline wander removed (Butterworth HPF, cutoff=%.1f Hz, order=%d).\n\n', ...
    HP_CUTOFF, HP_ORDER);

%% STEP 4 : TEMPLATE EXTRACTION
% Find the dominant R-peak in the search window (assumed Normal Sinus beat).
% Template = TEMPLATE_HALF samples each side of the R-peak (~0.222 s total).

idx_search_start = max(round(SEARCH_START * Fs), TEMPLATE_HALF + 1);
idx_search_end   = min(round(SEARCH_END   * Fs), N - TEMPLATE_HALF);

[r_peak_val, r_local_idx] = max(ecg_filtered(idx_search_start : idx_search_end));
r_peak_idx = idx_search_start + r_local_idx - 1;

template      = ecg_filtered((r_peak_idx - TEMPLATE_HALF) : (r_peak_idx + TEMPLATE_HALF));
template_time = (-TEMPLATE_HALF : TEMPLATE_HALF) / Fs;   % [s], centered at 0

fprintf('Template extracted: R-peak at sample #%d (t=%.4f s), amplitude=%.4f mV\n', ...
    r_peak_idx, time_raw(r_peak_idx), r_peak_val);
fprintf('Template length: %d samples (%.4f s)\n\n', length(template), length(template)/Fs);

%% STEP 5 : CROSS-CORRELATION (TEMPLATE MATCHING)
% Compute raw xcorr then normalize manually so scores are bounded in [-1,1].
% 'coeff' normalization: r[k] = R_xy[k] / sqrt(E_x_segment * E_template).
% This makes the threshold amplitude-invariant.

[xcorr_raw, lags] = xcorr(ecg_filtered, template, 'none');

template_energy = sum(template .^ 2);
M = length(template);

xcorr_full = zeros(size(xcorr_raw));
for ki = 1 : length(lags)
    lag_k   = lags(ki);
    x_start = max(1, 1 + lag_k);
    x_end   = min(N, M + lag_k);
    if x_start > x_end; continue; end
    local_energy = sum(ecg_filtered(x_start : x_end) .^ 2);
    denom = sqrt(local_energy * template_energy);
    if denom > 0
        xcorr_full(ki) = xcorr_raw(ki) / denom;
    end
end

% Align the xcorr output so index k corresponds to the ECG at sample k
lag_zero_idx      = find(lags == 0);
xcorr_valid_start = max(lag_zero_idx - TEMPLATE_HALF, 1);
xcorr_valid_end   = min(lag_zero_idx - TEMPLATE_HALF + N - 1, length(xcorr_full));
xcorr_aligned     = xcorr_full(xcorr_valid_start : xcorr_valid_end);

% Pad / trim to exactly N samples for 1-to-1 alignment with the ECG
if length(xcorr_aligned) < N
    xcorr_aligned(end+1:N) = 0;
elseif length(xcorr_aligned) > N
    xcorr_aligned = xcorr_aligned(1:N);
end

fprintf('Cross-correlation computed: %d aligned samples.\n\n', N);

%% STEP 6 : BEAT DETECTION & CLASSIFICATION
% Detect R-peaks in the filtered ECG, then read the xcorr score at each peak.
% score >= CORR_THRESH -> Normal | score < CORR_THRESH -> Arrhythmia / PVC

ecg_range     = max(ecg_filtered) - min(ecg_filtered);
min_dist_samp = round(MIN_BEAT_SEP * Fs);

[~, peak_locs] = findpeaks(ecg_filtered, ...
    'MinPeakProminence', 0.25 * ecg_range, ...
    'MinPeakDistance',   min_dist_samp);

num_beats       = length(peak_locs);
corr_scores     = xcorr_aligned(peak_locs);
is_normal       = corr_scores >= CORR_THRESH;
is_arrhythmia   = ~is_normal;
normal_locs     = peak_locs(is_normal);
arrhythmia_locs = peak_locs(is_arrhythmia);
num_normal      = sum(is_normal);
num_arrhythmia  = sum(is_arrhythmia);

fprintf('QRS peaks detected   : %d\n', num_beats);
fprintf('Normal (r >= %.2f)  : %d\n', CORR_THRESH, num_normal);
fprintf('Arrhythmia (r < %.2f): %d\n\n', CORR_THRESH, num_arrhythmia);

% Estimate heart rate from Normal beat RR intervals
if num_normal >= 2
    avg_rr       = mean(diff(normal_locs) / Fs);
elseif num_beats >= 2
    avg_rr       = mean(diff(peak_locs) / Fs);
else
    avg_rr       = 0.83;
end
detected_bpm = 60 / avg_rr;
fprintf('Estimated heart rate : %.1f BPM (avg RR = %.3f s)\n\n', detected_bpm, avg_rr);

%% STEP 7 : VISUALIZATION (4-Panel Figure)

fig = figure('Name', 'ECG Arrhythmia Detection - Cross-Correlation', ...
    'NumberTitle', 'off', 'Color', 'white', 'Position', [50, 10, 1200, 1200]);

ecg_color   = [0.12, 0.47, 0.71];
norm_color  = [0.17, 0.63, 0.17];
arrh_color  = [0.84, 0.15, 0.16];
xcorr_color = [0.58, 0.40, 0.74];
font_name   = 'Helvetica';

% --- Subplot 1: Normal QRS Template ---
ax1 = subplot(4, 1, 1);

plot(template_time * 1000, template, 'Color', ecg_color, 'LineWidth', 1.8);
hold on;
plot(0, template(TEMPLATE_HALF + 1), 'o', ...
    'MarkerFaceColor', arrh_color, 'MarkerEdgeColor', arrh_color, 'MarkerSize', 7);
y_pad = [min(template)-0.05, max(template)+0.05];
fill([template_time(1)*1000 template_time(end)*1000 template_time(end)*1000 template_time(1)*1000], ...
     [y_pad(1) y_pad(1) y_pad(2) y_pad(2)], ...
     [0.90 0.95 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'HandleVisibility', 'off');
hold off; grid on;

title(sprintf('Subplot 1: Normal QRS Template  |  t_{R} = %.3f s  |  %d samples', ...
    time_raw(r_peak_idx), length(template)), 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Time relative to R-peak (ms)'); ylabel('Amplitude (mV)');
legend({'Normal QRS Morphology', 'R-peak Anchor'}, 'FontSize', 9, ...
    'Location', 'northeastoutside', 'Box', 'off');
set(ax1, 'FontSize', 9, 'FontName', font_name, 'Box', 'off', 'GridAlpha', 0.2);
xlim([template_time(1)*1000, template_time(end)*1000]);

% --- Subplot 2: Filtered ECG with beat markers ---
ax2 = subplot(4, 1, 2);

plot(time_raw, ecg_filtered, 'Color', ecg_color, 'LineWidth', 0.9, ...
    'DisplayName', 'Filtered ECG (HPF 0.5 Hz)');
hold on;

if ~isempty(normal_locs)
    scatter(time_raw(normal_locs), ecg_filtered(normal_locs), 60, 'o', ...
        'MarkerEdgeColor', norm_color, 'MarkerFaceColor', norm_color, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Normal (%d)', num_normal));
end
if ~isempty(arrhythmia_locs)
    scatter(time_raw(arrhythmia_locs), ecg_filtered(arrhythmia_locs), 100, 'x', ...
        'MarkerEdgeColor', arrh_color, 'LineWidth', 2.0, ...
        'DisplayName', sprintf('Arrhythmia / PVC (%d)', num_arrhythmia));
end
xline(time_raw(r_peak_idx), '--', 'Template', 'Color', [0.5 0.5 0.5], ...
    'LineWidth', 1.0, 'FontSize', 7, 'HandleVisibility', 'off');
hold off; grid on;

title(sprintf('Subplot 2: Filtered ECG  |  Normal: %d  |  Arrhythmia: %d', ...
    num_normal, num_arrhythmia), 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Amplitude (mV)');
legend('FontSize', 9, 'Location', 'bestoutside', 'Box', 'off');
set(ax2, 'FontSize', 9, 'FontName', font_name, 'Box', 'off', 'GridAlpha', 0.2);
xlim([time_raw(1), time_raw(end)]);

% --- Subplot 3: Normalized cross-correlation score ---
ax3 = subplot(4, 1, 3);

plot(time_raw, xcorr_aligned, 'Color', xcorr_color, 'LineWidth', 1.2, ...
    'DisplayName', 'Normalized Xcorr (r)');
hold on;

curr_ylim = [min(-0.2, min(xcorr_aligned)-0.05), 1.05];
fill([time_raw(1); time_raw(end); time_raw(end); time_raw(1)], ...
     [curr_ylim(1); curr_ylim(1); CORR_THRESH; CORR_THRESH], ...
     [1.0 0.88 0.88], 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'HandleVisibility', 'off');
yline(CORR_THRESH, '--', sprintf('Threshold = %.2f', CORR_THRESH), ...
    'Color', arrh_color, 'LineWidth', 1.5, 'FontSize', 9, 'HandleVisibility', 'off');

if ~isempty(normal_locs)
    scatter(time_raw(normal_locs), xcorr_aligned(normal_locs), 50, 'o', ...
        'MarkerEdgeColor', norm_color, 'MarkerFaceColor', norm_color, 'LineWidth', 1.5, ...
        'DisplayName', 'Normal score (r >= thresh)');
end
if ~isempty(arrhythmia_locs)
    scatter(time_raw(arrhythmia_locs), xcorr_aligned(arrhythmia_locs), 80, 'x', ...
        'MarkerEdgeColor', arrh_color, 'LineWidth', 2.0, ...
        'DisplayName', 'Arrhythmia score (r < thresh)');
end

% Compact annotation + drop line for each arrhythmia beat
for i = 1 : num_arrhythmia
    ax_t = time_raw(arrhythmia_locs(i));
    ax_y = xcorr_aligned(arrhythmia_locs(i));
    text(ax_t + 0.05, ax_y, sprintf('r=%.2f', ax_y), ...
        'HorizontalAlignment', 'left', 'FontSize', 6, ...
        'Color', arrh_color, 'FontWeight', 'bold', 'FontName', font_name);
    plot([ax_t, ax_t], [ax_y, CORR_THRESH], ':', ...
        'Color', arrh_color, 'LineWidth', 1.0, 'HandleVisibility', 'off');
end
hold off; grid on;

title(sprintf('Subplot 3: Cross-Correlation Score  |  Threshold = %.2f', CORR_THRESH), ...
    'FontSize', 11, 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Correlation Coefficient (r)');
legend('FontSize', 9, 'Location', 'bestoutside', 'Box', 'off');
set(ax3, 'FontSize', 9, 'FontName', font_name, 'Box', 'off', 'GridAlpha', 0.2);
xlim([time_raw(1), time_raw(end)]); ylim(curr_ylim);

% --- Subplot 4: Synthetic Normal Sinus Rhythm (Gaussian PQRST model) ---
% Gaussian-kernel PQRST waveform for visual comparison at detected BPM.
ax4 = subplot(4, 1, 4);

samples_per_beat = round(avg_rr * Fs);
bt       = (0 : samples_per_beat-1) / Fs;
r_center = 0.45 * avg_rr;
gauss    = @(t, amp, mu, sig) amp * exp(-((t-mu).^2) / (2*sig^2));

single_beat = gauss(bt, 0.20, r_center-0.16, 0.025) ...   % P wave
            + gauss(bt,-0.07, r_center-0.03, 0.006) ...   % Q wave
            + gauss(bt, 1.00, r_center,      0.008) ...   % R wave
            + gauss(bt,-0.12, r_center+0.03, 0.006) ...   % S wave
            + gauss(bt, 0.30, r_center+0.25, 0.050);      % T wave
single_beat(1) = 0; single_beat(end) = 0;

synth_ecg  = zeros(1, N);
synth_time = (0 : N-1) / Fs;
beat_start = 1;
while beat_start + samples_per_beat - 1 <= N
    synth_ecg(beat_start : beat_start + samples_per_beat - 1) = single_beat;
    beat_start = beat_start + samples_per_beat;
end
remaining = N - beat_start + 1;
if remaining > 0
    synth_ecg(beat_start:end) = single_beat(1:remaining);
end
synth_ecg = synth_ecg * (r_peak_val / max(abs(single_beat)));

synth_color = [0.13 0.55 0.13];
plot(synth_time, synth_ecg, 'Color', synth_color, 'LineWidth', 1.2, ...
    'DisplayName', sprintf('Synthetic NSR (%.0f BPM)', detected_bpm));
hold on;
[synth_pks, synth_pk_locs] = findpeaks(synth_ecg, ...
    'MinPeakHeight',    0.5 * r_peak_val, ...
    'MinPeakDistance',  round(0.5 * samples_per_beat));
scatter(synth_time(synth_pk_locs), synth_pks, 40, 'o', ...
    'MarkerEdgeColor', synth_color, 'MarkerFaceColor', synth_color, ...
    'DisplayName', 'R-peaks (synthetic)');
hold off; grid on;

title(sprintf('Subplot 4: Ideal NSR Reference  |  %.0f BPM  |  RR = %.3f s', ...
    detected_bpm, avg_rr), 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Amplitude (mV)');
legend('FontSize', 9, 'Location', 'bestoutside', 'Box', 'off');
set(ax4, 'FontSize', 9, 'FontName', font_name, 'Box', 'off', 'GridAlpha', 0.2);
xlim([synth_time(1), synth_time(end)]);

sgtitle({'ECG Arrhythmia Detection via Cross-Correlation Template Matching', ...
         sprintf('MIT-BIH Record %s | Fs = 360 Hz | Detected HR = %.0f BPM', ...
         record_name, detected_bpm)}, 'FontSize', 13, 'FontWeight', 'bold');

%% STEP 8 : SAVE FIGURE

output_filename = 'ecg_arrhythmia_detection_result.png';
fprintf('Saving figure as "%s" at %d DPI...\n', output_filename, FIGURE_DPI);
try
    exportgraphics(fig, output_filename, 'Resolution', FIGURE_DPI);
catch
    print(fig, output_filename, '-dpng', sprintf('-r%d', FIGURE_DPI));
end
fprintf('Figure saved.\n\n');

%% STEP 9 : RESULTS SUMMARY

fprintf('--- DETECTION RESULTS SUMMARY ---\n');
fprintf('Total QRS beats       : %d\n', num_beats);
fprintf('Normal (r >= %.2f)   : %d\n', CORR_THRESH, num_normal);
fprintf('Arrhythmia (r < %.2f): %d\n\n', CORR_THRESH, num_arrhythmia);

for k = 1 : num_beats
    if is_normal(k), label = 'NORMAL';
    else,            label = 'ARRHYTHMIA';
    end
    fprintf('Beat #%2d | t=%5.3f s | r=%+.4f | %s\n', ...
        k, time_raw(peak_locs(k)), corr_scores(k), label);
end

fprintf('\n[DONE] Script completed. Output: %s\n', output_filename);

