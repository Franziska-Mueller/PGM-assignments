%   Instructional material for COM-711 Selected Topics in Computer Vision
%   Autumn 2011
%
%   Assignment #5 - Particle filter
%
%   Copyright  2011 Computer Vision Lab, 
%   Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Author:    Kevin Smith         http://www.kev-smith.com
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
%   This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.

function tracker()

% Read the video sequence into videoSeq as a cell 
% NOTE: CHANGE FILETYPE AS APPROPRIATE FOR EACH SEQUENCE (*.png, *.bmp, or *.jpg)
imgPath = '../assignment04-data/sequence3/'; dCell = dir([imgPath '*.jpg']);
disp('Loading image files from the video sequence, please be patient.');
for d = 1:length(dCell)
    videoSeq{d} = imread([imgPath dCell(d).name]);
end

% Define a path to save your results images and make sure it exists
resultsPath = './results3/';
if ~isdir(resultsPath)
    mkdir(resultsPath);
end

% Create a figure we will use to display the video sequence
clf; figure(1); set(gca, 'Position', [0 0 1 1]);
imshow(videoSeq{1});

% You might need to know the size of video frames 
% (i.e. to constrain the state so it doesn't wander outside of the video).
VIDEO_WIDTH = size(videoSeq{1},2);  VIDEO_HEIGHT = size(videoSeq{1},1);


% Because we don't have an automatic detection method to initialize the
% tracker, we will do it by hand. A GUI interface allows us to select a 
% bouding box from the first frame of the sequence.
display('Initialize the bounding box by cropping the face.');
[patch, bB] = imcrop();
bB = round(bB);

% Fill the initial state vector with our hand initialization
x_init = [bB(1) bB(2) 0 0 bB(3)/bB(4) bB(4)];

% Draw a bouding box on the image showing our initial state estimate
drawBox(x_init);

% TODO: Here you can load or initialize your color model
xHistogram = LABhistogram(patch);

% TODO: Here you can define your particle set.  There are many possible
% ways to implement it, I used a Matlab structure.  Remember each particle
% consists of a state estimate and an associated weight.
N = 150;    % the number of samples (particles) to use
mu = x_init(1:2);
sigma = [5 0; 0 5];
xSamples = cell(N, 2);
xSamples{1,1} = x_init;
xSamples{1,2} = observationModel(xSamples{1,1}, videoSeq{1}, xHistogram, VIDEO_WIDTH, VIDEO_HEIGHT);
newSamples = cell(N,2);

for i = 2:N
    noise = mvnrnd(mu,sigma);
    xSamples{i, 1} = [noise(1) noise(2) 0 0 x_init(5) x_init(6)];
    xSamples{i, 2} = observationModel(xSamples{i,1}, videoSeq{1}, xHistogram, VIDEO_WIDTH, VIDEO_HEIGHT);
end 

% MAIN LOOP: we loop through the video sequence and estimate the state at
% each time step, t

T = length(videoSeq);
for t=1:T

    % load the current image and display it in the figure
    I = videoSeq{t}; figure(1); cla; imshow(I); hold on;
    
    % TODO: Insert your code here for prediction, updating, reweighting,
    % resamping, etc.
    
    % resample
    sumWeightsInv = 1/sum(cell2mat(xSamples(:,2)));
    for i=1:N
       u = random('Uniform',0,1); 
       com = 0;
       for j=1:N
        com = com + xSamples{j, 2} * sumWeightsInv;
        if u <= com
            newSamples{i, 1} = motionPredict(xSamples{j, 1}, VIDEO_WIDTH, VIDEO_HEIGHT);
            newSamples{i, 2} = observationModel(newSamples{i, 1}, I, xHistogram, VIDEO_WIDTH, VIDEO_HEIGHT);
            break;
        end
       end
    end
    xSamples = newSamples;
    
    
    
    % TODO: Once you have performed all particle filtering steps, you
    % still must infer a solution from the approximate distribution.  I
    % simply took the estimate as the mean state vector computed from all
    % particles.
    
    % mean
    % estimate_t = mean(cell2mat(xSamples(:, 1)));
    % weighted mean
    sumWeightsInv = 1/sum(cell2mat(xSamples(:,2)));
    estimate_t = sumWeightsInv * sum(repmat(cell2mat(xSamples(:,2)),1,6) .* cell2mat(xSamples(:,1)));
    
    % draw the estimated state for time t to the figure and save the
    % figure to a file in your results directory (UNCOMMENT THE PRINT
    % STATEMENT TO WRITE FIGURES TO AN IMAGE WHICH YOU CAN ENCODE INTO A MOVIE FILE)
    for i = 1:N 
        drawBox(xSamples{i, 1}, [0 0 1]);
    end
    drawBox(estimate_t, [0 1 0]);
    %print(gcf, '-dpng', [resultsPath 'f' number_into_string(t,1000) '.png'], '-r100');
    
    
    % allow the figure to refresh so we can see our results
    pause(0.01); refresh;
    
end
   


end





%========================== SUPPLEMENTARY FUNCTIONS ======================

% Draws a bounding box given a state estimate.  Color of the bounding box
% can be given as an optional second argument: drawbox(x_t, [0 1 0]) will
% give a green bounding box.
function drawBox(x, varargin)
r = x(1);
c = x(2);
w = x(6)*x(5);
h = x(6);

x1 = r;
x2 = r + w;
y1 = c;
y2 = c + h;

if nargin == 1
    line([x1 x1 x2 x2 x1], [y1 y2 y2 y1 y1]);
else
    line([x1 x1 x2 x2 x1], [y1 y2 y2 y1 y1], 'Color', varargin{1}, 'LineWidth', 0.5);
end

end


% OBSERVATION MODEL.  Computes the likelihood that data observed in the
% image I (z_t) supports the state estimated by x_t.  I computed the
% likelihood by comparing a color histogram extracted from the bounding box
% defined by x_to to a known color model which I extracted beforehand. I
% modeled p(z_t|x_t) \propto exp(-lambda*dist(h, colormodel)) where dist
% is the KL divergence of two histograms, h is a color histogram
% corresponding to x_t, colorModel is a known color model, and lambda is a
% hyperparameter used to adjust the pickiness of the likelihood function.
% You may, however, use any likelihood model you prefer.
function z_t = observationModel(x_t, I, colorModel, VIDEO_WIDTH, VIDEO_HEIGHT)

% I found this value of lambda to work well
lambda = 15;

% You might find image patch corresponding to the x_t bounding box useful
r = round(x_t(2)); c = round(x_t(1));
w = round(x_t(5)*x_t(6)); h = round(x_t(6));
r2 = min(VIDEO_HEIGHT, r+h+1);
c2 = min(VIDEO_WIDTH, c+w+1);
imagePatch = I(r:r2, c:c2,:);

patchHistrogram = LABhistogram(imagePatch);

% TODO: place your code to compute the likelihood z_t here
z_t = exp(-lambda*KLdivergence(patchHistrogram, colorModel));
end


% MOTION PREDICTION MODEL. Given a past state vector x_t1, predicts a new 
% state vector x_t according to the motion model.
function x_t = motionPredict(x_t1, VIDEO_WIDTH, VIDEO_HEIGHT)

% You might find that constraining certain elements of the state such as
% the bounding box height or aspect ratio can improve your tracking
MIN_h = 20; 
MAX_h = 40; 
%MIN_h = 100;
%MAX_h = 200;
%MIN_a = .66;
MIN_a = 0.4;
MAX_a = min(1.75,VIDEO_WIDTH/MAX_h);   

% TODO: Define your motion prediction model, including F and Q, and apply it
% to predict a new state x_t given the previous state x_t1.  Remember to
% handle the boundary conditions where the state estimate reaches the 
% edges of the video sequence.


sigma_xpos = max(1, 0.1*abs(x_t1(3)));
sigma_ypos = max(1, 0.1*abs(x_t1(4)));
sigma_xvel = 2.5;
sigma_yvel = 2.5;
sigma_a = 1;
sigma_h = 3;

rho = 0.3;
covMat = [sigma_xpos^2, rho*sigma_xpos*sigma_ypos, 0, 0, 0, 0; % positions
          rho*sigma_xpos*sigma_ypos, sigma_ypos^2, 0, 0, 0, 0;
          0, 0, sigma_xvel^2, rho*sigma_xvel*sigma_yvel, 0, 0; % velocity
          0, 0, rho*sigma_xvel*sigma_yvel, sigma_yvel^2, 0, 0;
          0, 0, 0, 0, sigma_a^2, 0; % bounding box size
          0, 0, 0, 0, 0, sigma_h^2;
        ];
sample = mvnrnd(zeros(1,6), covMat);
x_t = x_t1;
x_t(1:2) = [x_t1(1)+x_t1(3) x_t1(2)+x_t1(4)];
x_t = x_t + sample;

x_t(6) = min(max(x_t(6), MIN_h), MAX_h);
x_t(5) = min(max(x_t(5), MIN_a), MAX_a);
w = x_t(5) * x_t(6);
x_t(1) = min(max(1, x_t(1)), VIDEO_WIDTH - w);
x_t(2) = min(max(1, x_t(2)), VIDEO_HEIGHT - x_t(6));
if x_t(1) < 0
    a = 0;
end
end


% KL DIVERGENCE Computes the KL divergence (a distance measure) for two 
% 1-D histograms. 
function k = KLdivergence(H1n,H2n)

eta = .00001 * ones(size(H1n));
H1n = H1n+eta;
H2n = H2n+eta;
temp = H1n.*log(H1n./H2n);
temp(isnan(temp)) = 0;
k = sum(temp);
end



