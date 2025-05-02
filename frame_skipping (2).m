clear all
close all
clc
%% Preprocessing Step 
% Run the file in video folder
x = dir('*.avi');
[length temp] = size(x);
for k=1:length
    videoname = x(k).name;
    videoFileReader = vision.VideoFileReader(videoname);

    numFrames = 0;
    while ~isDone(videoFileReader)
        step(videoFileReader);
        numFrames = numFrames + 1;
    end

    reset(videoFileReader); 

    % Process all frames in the video
    movMean = step(videoFileReader);
    imgB = movMean;
    imgBp = imgB;
    correctedMean = imgBp;
  range = 30:30:numFrames;
    Hcumulative = eye(3);
    for i=1:size(range,2)
        ii=range(i);
        ref=ii;
        while ~isDone(videoFileReader) && ii < ref + 30
            % Read in new frame
            Frame1 = imgB; % z^-1
            imgAp = imgBp; % z^-1
            imgB = step(videoFileReader);
            movMean = movMean + imgB;
            ii = ii+1;
        end
        imwrite(Frame1,strcat( string(k), string(i),'.jpg'));
    end
end