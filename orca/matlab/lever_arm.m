function [phase_center] = lever_arm(param, tx_weights, rxchannel)
% lever_arm  ORCA stub returning zero phase centers.
%
% PLACEHOLDER — DO NOT USE FOR REAL PROCESSING.
%
% Returns a 3-by-numel(rxchannel) matrix of zeros, representing the radar
% phase center(s) in the GPS-antenna body frame [m]:
%   row 1 = x  (forward / along-track)
%   row 2 = y  (left / cross-track)
%   row 3 = z  (up)
%
% This lets `records_create` and downstream stages run end-to-end against
% real ORCA data so the pipeline can be exercised before the antenna /
% IMU / GPS geometry has been measured. The GPS-antenna, IMU, and radar
% phase-center offsets are all conflated to zero; results from any stage
% that depends on lever-arm-corrected positions (SAR, motion comp,
% along-track decimation) will be incorrect until this is replaced with
% measured values.
%
% Wiring: this function is selected via the radar.lever_arm_fh column
% in the parameter spreadsheet (currently `@lever_arm`, set in
% orca/seasons_config/glass_orca_season.yaml). Add orca/matlab/ to your
% MATLAB path *before* the main OPR path so this shadows OPR's
% lever_arm.m for ORCA seasons only, OR copy/merge this case into the
% main lever_arm.m and remove this stub.
%
% INPUTS
%   param        OPR parameter struct (used by real lever_arm.m to
%                switch on radar_name / season_name; unused here).
%   tx_weights   Vector of TX weights (unused here).
%   rxchannel    Vector of receive channel indices to return phase
%                centers for.
%
% OUTPUT
%   phase_center 3-by-numel(rxchannel) matrix of zeros.

phase_center = zeros(3, numel(rxchannel));

end
