function [output_index] = findStimStartOffline(input_code_array)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
% stimulus_code_array has shape (n, 1)
[n_length, ~] = size(input_code_array);

% compute the difference
input_code_diff = input_code_array(2:end, :) - input_code_array(1:(n_length-1),:);
% find all the indices where diff > 0, notice that the index
% needs to plus 1 for consistency.
output_index = find(input_code_diff > 0) + 1;
end

