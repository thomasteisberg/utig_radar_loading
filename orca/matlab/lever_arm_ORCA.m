function [phase_center] = lever_arm_ORCA(param, tx_weights, rxchannel)
% lever_arm_ORCA  ORCA stub returning zero phase centers.
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
% Wiring: selected via `radar.lever_arm_fh = @lever_arm_ORCA` (set in
% orca/seasons_config/glass_orca_season.yaml). Add orca/matlab/ to the
% MATLAB path so OPR can resolve the function handle. The name is
% intentionally not `lever_arm` so it doesn't shadow OPR's central
% lever_arm.m for UTIG/other seasons.
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
