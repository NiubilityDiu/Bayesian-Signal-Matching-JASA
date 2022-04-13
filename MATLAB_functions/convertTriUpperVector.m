function [TriuMatArr] = convertTriUpperVector(TriuMat)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[~, n] = size(TriuMat);
TriuMatArr = zeros(n*(n+1)/2 , 1);
index = 0;
for i=1:n
    % disp(index);
    TriuMatArr((index+1):(index+n-i+1), :) = TriuMat(i, i:n);
    index = index + n - i + 1;
end
end