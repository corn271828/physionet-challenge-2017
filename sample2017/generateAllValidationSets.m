clear all;
close all;
clc
%directories = {'validation' 'validation-mcar' 'validation-0' 'validation-mean' 'validation-locf'};
directories = {'validation-locf'};
for i = 1:length(directories)
    current_directory = directories{i};
    generateValidationSet(current_directory);
end